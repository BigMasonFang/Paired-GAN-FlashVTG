import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_mask_1d(center, width, seq_len, device):
    """
    Generates a differentiable 1D Gaussian mask.
    Args:
        center: Tensor (B,) normalized center [0, 1]
        width: Tensor (B,) normalized width [0, 1]
        seq_len: int, length of the sequence
    """
    # Create a time grid (B, T)
    grid = torch.linspace(0, 1, seq_len, device=device).unsqueeze(0).repeat(center.size(0), 1)
    
    # Gaussian formula: exp(- (x - mu)^2 / (2 * sigma^2))
    # We treat width as 2*sigma (approx 95% of mass)
    sigma = width / 4.0 + 1e-6 # Avoid div by zero
    
    mask = torch.exp(-((grid - center.unsqueeze(1)) ** 2) / (2 * sigma.unsqueeze(1) ** 2))
    return mask


def binary_mask_1d(center, width, seq_len, device):
    """Generates a differentiable-ish binary mask (Soft Step)."""
    grid = torch.linspace(0, 1, seq_len, device=device).unsqueeze(0).repeat(center.size(0), 1)
    
    # Calculate start and end points
    start = center - width / 2
    end = center + width / 2
    
    # Sigmoid approximation of a step function for differentiability
    # Steepness factor k
    k = 50.0 
    mask = torch.sigmoid(k * (grid - start.unsqueeze(1))) * torch.sigmoid(k * (end.unsqueeze(1) - grid))
    return mask


class MultiHeadAttentionDiscriminator(nn.Module):
    """Discriminator with multi-head attention and temporal convolution"""

    def __init__(self, input_dim=2818, hidden_dim=512, num_heads=4, text_dim=0):
        super().__init__()
        combined_dim = input_dim + text_dim
        self.input_proj = nn.Linear(combined_dim, hidden_dim)

        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim,
                                               num_heads=num_heads,
                                               batch_first=True)

        # Temporal convolution branch
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2), nn.BatchNorm1d(hidden_dim * 2),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2))

        # Final discrimination head (No Sigmoid for WGAN!)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, 256), nn.ReLU(),
                                nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, x, txt_emb=None):
        # x: (B, T, V_dim)
        # txt_emb: (B, T_dim) -> We need to expand this to (B, T, T_dim)
        
        if txt_emb is not None:
            # Expand text to match video sequence length
            seq_len = x.size(1)
            txt_expanded = txt_emb.unsqueeze(1).repeat(1, seq_len, 1) # (B, T, T_dim)
            
            # Concatenate along feature dimension
            x = torch.cat([x, txt_expanded], dim=-1) # (B, T, V_dim + T_dim)

        # Input projection
        x = self.input_proj(x)  # (B, T, D_hid)

        # Attention pathway
        attn_out, _ = self.attention(x, x, x)

        # Convolution pathway
        conv_out = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Combine features
        combined = torch.cat([attn_out.mean(1), conv_out.mean(1)], dim=1)

        return self.fc(combined)


class ConvLSTMDiscriminator(nn.Module):
    """Combines 1D convolutions with LSTM for temporal analysis"""

    def __init__(self, input_dim=2818, conv_dim=512, lstm_dim=256):
        super().__init__()
        # Convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2), nn.BatchNorm1d(conv_dim), nn.MaxPool1d(2),
            nn.Conv1d(conv_dim, conv_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2), nn.BatchNorm1d(conv_dim // 2))

        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size=conv_dim // 2,
                            hidden_size=lstm_dim,
                            bidirectional=True,
                            batch_first=True)

        # Attention-based pooling
        self.attention = nn.Sequential(nn.Linear(lstm_dim * 2, 128), nn.Tanh(),
                                       nn.Linear(128, 1), nn.Softmax(dim=1))

        # Final classification (No Sigmoid for WGAN!)
        self.fc = nn.Sequential(nn.Linear(lstm_dim * 2, 1))

    def forward(self, x):
        # Conv processing
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T//2, D_conv)

        # LSTM processing
        lstm_out, _ = self.lstm(x)  # (B, T//2, D_lstm*2)

        # Attention pooling
        attn_weights = self.attention(lstm_out)
        context = torch.sum(lstm_out * attn_weights, dim=1)

        return self.fc(context)


class TemporalDiscriminator(nn.Module):
    """
    Hybrid Discriminator: Gating (Early Fusion) + Projection (Late Fusion).
    """
    def __init__(self, input_dim=2818, text_dim=0, hidden_dim=256):
        super().__init__()
        
        # --- EARLY FUSION (Gating) Components ---
        self.video_proj = nn.Linear(input_dim, hidden_dim)
        
        # Text Gate: Produces a 0-1 mask to filter video
        self.text_gate = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid() 
        )
        
        # --- CORE Processing ---
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 
                            bidirectional=True, batch_first=True)
        
        # --- LATE FUSION (Projection) Components ---
        # 1. Unconditional Head (Judges "Is this a realistic highlight?")
        self.linear_u = nn.Linear(2*hidden_dim, 1)
        
        # 2. Conditional Projection (Judges "Does this match the text?")
        self.text_proj_late = nn.Linear(text_dim, 2*hidden_dim) 

    def forward(self, x, txt_emb=None):
        # x: (B, T, V_dim)
        # txt_emb: (B, T_dim)
        
        # === STEP 1: EARLY FUSION (Gating) ===
        # Project Video
        x_feat = self.video_proj(x) # (B, T, H)
        
        if txt_emb is not None:
            # Create Gate
            gate = self.text_gate(txt_emb).unsqueeze(1) # (B, 1, H)
            # Filter Video: "Only look at parts relevant to text"
            x_feat = x_feat * gate 

        # === STEP 2: TEMPORAL PROCESSING ===
        # Disable CuDNN for double-backprop safety (WGAN support)
        with torch.backends.cudnn.flags(enabled=False):
            self.lstm.flatten_parameters()
            lstm_out, _ = self.lstm(x_feat)
            
        # Global video representation (Max Pool)
        v_feat = torch.max(lstm_out, dim=1)[0] # (B, 2H)
        
        # === STEP 3: LATE FUSION (Projection) ===
        # Base Score (Realism)
        out = self.linear_u(v_feat)
        
        if txt_emb is not None:
            # Project text to shared space
            t_feat_late = self.text_proj_late(txt_emb) # (B, 2H)
            
            # Dot Product Alignment (Projection)
            # "Does the global video feature align with the text feature?"
            projection = (v_feat * t_feat_late).sum(dim=1, keepdim=True)
            
            # Final Score = Realism + Alignment
            out = out + projection
            
        return out


class FlashVTGWithGAN(nn.Module):
    def __init__(self, flash_vtg, discriminator, mask_type='gaussian', mix_saliency=True):
        super().__init__()
        self.flash_vtg = flash_vtg
        self.discriminator = discriminator
        self.mask_type = mask_type       # 'gaussian' or 'binary'
        self.mix_saliency = mix_saliency # True or False

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, vid, qid, targets=None):
        # 1. Forward Generator
        output = self.flash_vtg(src_txt, src_txt_mask, src_vid, src_vid_mask, vid, qid, targets)

        # 2. Extract Predicted Spans
        if "out_coord" in output and "out_class" in output and "point" in output:
            out_coord = output["out_coord"] 
            out_class = output["out_class"] 
            point = output["point"]         

            # Fix dimensions if needed (Robustness)
            if out_class.dim() == 3 and out_class.size(1) == 1: 
                out_class = out_class.permute(0, 2, 1) 
            if out_coord.dim() == 3 and out_coord.size(1) == 2:
                out_coord = out_coord.permute(0, 2, 1) 
            if point.dim() == 2:
                point = point.unsqueeze(0).expand(out_class.size(0), -1, -1)

            # Select Best Span
            probs = torch.sigmoid(out_class).squeeze(-1)
            best_idx = torch.argmax(probs, dim=1)
            
            batch_indices = torch.arange(out_coord.size(0), device=out_coord.device)
            sel_coord = out_coord[batch_indices, best_idx]
            sel_point = point[batch_indices, best_idx]
            
            # Decode Span
            pred_center = (sel_coord[:, 0] * -1) * sel_point[:, 3] + sel_point[:, 0]
            pred_width  = torch.exp(sel_coord[:, 1]) * sel_point[:, 3]
            
            seq_len = src_vid.size(1) 
            
            # === OPTION 1: MASK TYPE ===
            if self.mask_type == 'binary':
                span_mask = binary_mask_1d(pred_center, pred_width, seq_len, src_vid.device)
            else: # default gaussian
                span_mask = gaussian_mask_1d(pred_center, pred_width, seq_len, src_vid.device)
                
            span_mask = span_mask.unsqueeze(-1) # (B, T, 1)

            # === OPTION 2: MIXING STRATEGY ===
            if self.mix_saliency:
                saliency = torch.sigmoid(output["saliency_scores"]).unsqueeze(-1)
                final_mask = saliency * span_mask 
            else:
                final_mask = span_mask # Span only
            
        else:
            # Fallback
            final_mask = (torch.sigmoid(output["saliency_scores"]) > 0.5).float().unsqueeze(-1)

        mask_expanded = src_txt_mask.unsqueeze(-1) # (B, L, 1)
        txt_sum = (src_txt * mask_expanded).sum(dim=1)
        txt_len = mask_expanded.sum(dim=1).clamp(min=1e-9)
        txt_pooled = txt_sum / txt_len # (B, D_txt)
        
        # output["fake_features"] = src_vid * final_mask
        output["txt_pooled"] = txt_pooled # Store this to use in train.py
        output["fake_features"] = src_vid * final_mask
        return output


def wgan_gp_mse_loss(d_real,
                     d_fake,
                     real_feats,
                     fake_feats,
                     discriminator,
                     lambda_gp=10,
                     lambda_mse=0.1,
                     txt_emb=None):
    """
    Combined loss for GAN training with:
    - Wasserstein Loss (WGAN-GP)
    - Gradient Penalty
    - Mean Squared Error (MSE)
    """
    # 1. Wasserstein Loss (original WGAN loss)
    wasserstein_loss = torch.mean(d_fake) - torch.mean(d_real)

    # 2. Gradient Penalty (WGAN-GP)
    alpha = torch.rand(real_feats.size(0), 1, 1, device=real_feats.device)
    interpolates = (alpha * real_feats +
                    (1 - alpha) * fake_feats).requires_grad_(True)

    if txt_emb is not None:
        d_interp = discriminator(interpolates, txt_emb=txt_emb)
    else:
        d_interp = discriminator(interpolates)
    
    # Handle discriminator output shape (B, 1) or (B,)
    d_interp = d_interp.view(d_interp.size(0), -1)
    
    gradients = torch.autograd.grad(outputs=d_interp,
                                    inputs=interpolates,
                                    grad_outputs=torch.ones_like(d_interp),
                                    create_graph=True,
                                    retain_graph=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=(1, 2)) - 1)**2).mean()

    # 3. MSE Loss (additional alignment - optional)
    mse_loss = F.mse_loss(d_real, torch.ones_like(d_real)) + \
               F.mse_loss(d_fake, torch.zeros_like(d_fake))

    # Combine all components
    total_loss = wasserstein_loss + lambda_gp * gradient_penalty + lambda_mse * mse_loss

    return total_loss, {
        'loss_wasserstein': wasserstein_loss.item(),
        'loss_gp': gradient_penalty.item(),
        'loss_mse': mse_loss.item()
    }


def feature_matching_loss(real_features, fake_features, lambda_fm=0.1):
    """
    Handles both 2D (B, D) and 3D (B, T, D) feature tensors
    """
    # Check dimensions and compute mean appropriately
    if real_features.dim() == 3:  # (B, T, D)
        real_mean = real_features.mean(dim=(1, 2))
        fake_mean = fake_features.mean(dim=(1, 2))
    elif real_features.dim() == 2:  # (B, D)
        real_mean = real_features.mean(dim=1)
        fake_mean = fake_features.mean(dim=1)
    else:
        raise ValueError(
            f"Unexpected feature dimension: {real_features.dim()}")

    fm_loss = F.mse_loss(fake_mean, real_mean)
    return lambda_fm * fm_loss, {'loss_fm': fm_loss.item()}