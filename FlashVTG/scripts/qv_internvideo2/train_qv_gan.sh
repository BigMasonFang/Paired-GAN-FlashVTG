dset_name=qv_internvideo2
ctx_mode=video_tef
v_feat_types=internvideo2
t_feat_type=llama
results_root=results_qv_gan
exp_id=demo

######## data paths
train_path=data/highlight_train_release_IV2.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
# video features
v_feat_dirs=/home/xiang_fang1/Desktop/flashVTG/data/highlight/intervideo2_video/qvhighlight_6b
v_feat_dim=768

# text features
t_feat_dir=/home/xiang_fang1/Desktop/flashVTG/data/highlight/llama_text/qvhighlight_llama_text_feature
t_feat_dim=4096

#### training config (from train.sh)
bsz=64
max_v_l=75
max_q_l=40 
eval_epoch=5
epoch=200
weight_decay=0.0001
eval_bsz=1

# Model Architecture params
enc_layers=3
t2v_layers=6
dummy_layers=2
num_dummies=40
kernel_size=5
num_conv_layers=1
num_mlp_layers=5

# Loss weights
lw_reg=1
lw_cls=5
lw_sal=0.1
lw_saliency=0.8
label_loss_coef=0

#### GAN Configuration (New additions)
add_gan=True
lambda_gan=0.05
gan_dis_type=0 # 1 is attention, 2 is LSTM+CNN, 0 is LSTM
gan_dis_loss=1 # 1 is wgan_gp_mse_loss, 2 is feature matching loss
mask_type=binary
mix_saliency=False

# Set seed (optional, can be passed or hardcoded)
seed=42

PYTHONPATH=$PYTHONPATH:. python FlashVTG/train.py \
data/MR_16.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--enc_layers ${enc_layers} \
--results_root ${results_root} \
--bsz ${bsz} \
--exp_id ${exp_id} \
--t2v_layers ${t2v_layers} \
--dummy_layers ${dummy_layers} \
--max_v_l ${max_v_l} \
--max_q_l ${max_q_l} \
--n_epoch ${epoch} \
--lr_drop 400 \
--eval_epoch ${eval_epoch} \
--wd ${weight_decay} \
--eval_bsz ${eval_bsz} \
--lw_reg ${lw_reg} \
--lw_cls ${lw_cls} \
--lw_sal ${lw_sal} \
--lw_saliency ${lw_saliency} \
--nms_thd 0.7 \
--use_neg \
--num_dummies ${num_dummies} \
--kernel_size ${kernel_size} \
--num_conv_layers ${num_conv_layers} \
--num_mlp_layers ${num_mlp_layers} \
--label_loss_coef ${label_loss_coef} \
--seed ${seed} \
--add_gan ${add_gan} \
--gan_dis_type ${gan_dis_type} \
--gan_dis_loss ${gan_dis_loss} \
--lambda_gan ${lambda_gan} \
--mask_type ${mask_type} \
--mix_saliency ${mix_saliency}\
${@:1}