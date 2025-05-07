import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionDiscriminator(nn.Module):
    """Discriminator with multi-head attention and temporal convolution"""

    def __init__(self, input_dim=2818, hidden_dim=512, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

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

        # Final discrimination head
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, 256), nn.ReLU(),
                                nn.Dropout(0.3), nn.Linear(256, 1),
                                nn.Sigmoid())

    def forward(self, x):
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

        # Final classification
        self.fc = nn.Sequential(nn.Linear(lstm_dim * 2, 1), nn.Sigmoid())

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
    """Discriminates real vs. predicted highlight clips."""

    def __init__(self, input_dim=2818, hidden_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim,
                            hidden_dim,
                            bidirectional=True,
                            batch_first=True)
        self.fc = nn.Sequential(nn.Linear(2 * hidden_dim, 1), nn.Sigmoid())

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        assert x.ndim == 3, f"Input must be 3D (batch, seq, feat). Got {x.shape}"
        assert x.size(-1) == self.input_dim, \
            f"Feature dim mismatch. Expected {self.input_dim}, got {x.size(-1)}"

        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Use last timestep


class FlashVTGWithGAN(nn.Module):

    def __init__(self, flash_vtg, discriminator):
        super().__init__()
        self.flash_vtg = flash_vtg
        self.discriminator = discriminator

    def forward(self,
                src_txt,
                src_txt_mask,
                src_vid,
                src_vid_mask,
                vid,
                qid,
                targets=None):
        # Original FlashVTG outputs
        output = self.flash_vtg(src_txt, src_txt_mask, src_vid, src_vid_mask,
                                vid, qid, targets)

        # Generate predicted highlight mask
        pred_scores = output["saliency_scores"]  # (batch, seq_len)
        mask = (pred_scores > 0.5).float().unsqueeze(-1)  # Threshold

        # Create fake features (mask non-highlight regions)
        fake_features = src_vid * mask  # (batch, seq_len, feat_dim)

        # Store for GAN loss calculation
        output["fake_features"] = fake_features
        return output


def wgan_gp_mse_loss(d_real,
                     d_fake,
                     real_feats,
                     fake_feats,
                     discriminator,
                     lambda_gp=10,
                     lambda_mse=0.1):
    """
    Combined loss for GAN training with:
    - Wasserstein Loss (WGAN-GP)
    - Gradient Penalty
    - Mean Squared Error (MSE)
    
    Args:
        d_real: Discriminator output for real features (unbounded)
        d_fake: Discriminator output for fake features (unbounded)
        real_feats: Ground truth features (B, T, D)
        fake_feats: Generated features (B, T, D)
        discriminator: The discriminator model
        lambda_gp: Weight for gradient penalty (default: 10)
        lambda_mse: Weight for MSE loss (default: 0.1)
    """
    # 1. Wasserstein Loss (original WGAN loss)
    wasserstein_loss = torch.mean(d_fake) - torch.mean(d_real)

    # 2. Gradient Penalty (WGAN-GP)
    alpha = torch.rand(real_feats.size(0), 1, 1, device=real_feats.device)
    interpolates = (alpha * real_feats +
                    (1 - alpha) * fake_feats).requires_grad_(True)

    d_interp = discriminator(interpolates)
    gradients = torch.autograd.grad(outputs=d_interp,
                                    inputs=interpolates,
                                    grad_outputs=torch.ones_like(d_interp),
                                    create_graph=True,
                                    retain_graph=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=(1, 2)) - 1)**2).mean()

    # 3. MSE Loss (additional alignment)
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
