dset_name=youtube_uni
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results_youtubeuni
exp_id=demo


######## data paths
# train_path=data/youtube_uni/youtube_train.jsonl
# eval_path=data/youtube_uni/youtube_anno.jsonl
train_path=data/youtube_uni/youtube_train.jsonl
eval_path=data/youtube_uni/youtube_valid.jsonl
eval_split_name=val

######## setup video+text features
# feat_root=/home/caozhuo/data_ssd/youtube_uni
# feat_root=/media/xiang/Slayer/flashVTG/data/youtube/youtube_uni
feat_root=/home/xiang_fang1/Desktop/flashVTG/data/youtube/youtube_uni

# # video features
v_feat_dim=2816
v_feat_dirs=()
v_feat_dirs+=(${feat_root}/vid_clip)
v_feat_dirs+=(${feat_root}/vid_slowfast)

# # text features
t_feat_dir=${feat_root}/txt_clip/ # maybe not used
t_feat_dim=512


#### training
bsz=4
lr=2e-4
enc_layers=3
t2v_layers=2
dummy_layers=2

kernel_size=5
num_conv_layers=2
num_mlp_layers=3

lw_cls=0.6
lw_sal=0.5
lw_saliency=0.7
label_loss_coef=5

epoch=150
max_es_cnt=20

lambda_gan=0.1
gan_dis_type=1 # 1 is attention, 2 is LSTM+CNN
# gan_dis_loss=1 # 1 is wgan_gp_mse_loss, 2 is feature matching loss
gan_dis_loss=2 # 1 is wgan_gp_mse_loss, 2 is feature matching loss

for num_dummies in 1
do 
    # for seed in 2024
    for seed in 10 42 100 1000
    do 
        # for dset_domain in gymnastics surfing
        # for dset_domain in gymnastics skating skiing parkour dog
        for dset_domain in gymnastics surfing skating skiing parkour dog
        # for dset_domain in surfing
        do
            PYTHONPATH=$PYTHONPATH:. python FlashVTG/train.py \
            data/HD.py \
            --dset_name ${dset_name} \
            --ctx_mode ${ctx_mode} \
            --train_path ${train_path} \
            --eval_path ${eval_path} \
            --eval_split_name ${eval_split_name} \
            --v_feat_dirs ${v_feat_dirs[@]} \
            --v_feat_dim ${v_feat_dim} \
            --t_feat_dir ${t_feat_dir} \
            --t_feat_dim ${t_feat_dim} \
            --bsz ${bsz} \
            --results_root ${results_root}/${dset_domain} \
            --exp_id ${exp_id} \
            --max_v_l 1000 \
            --n_epoch ${epoch} \
            --lr_drop 2000 \
            --max_es_cnt ${max_es_cnt} \
            --seed $seed \
            --lr ${lr} \
            --dset_domain ${dset_domain} \
            --enc_layers ${enc_layers} \
            --t2v_layers ${t2v_layers} \
            --dummy_layers ${dummy_layers} \
            --kernel_size ${kernel_size} \
            --num_conv_layers ${num_conv_layers} \
            --num_mlp_layers ${num_mlp_layers} \
            --clip_length 1 \
            --lw_cls ${lw_cls} \
            --lw_sal ${lw_sal} \
            --lw_saliency ${lw_saliency} \
            --label_loss_coef ${label_loss_coef} \
            --num_dummies ${num_dummies} \
            --num_workers 4 \
            --use_neg \
            --eval_bsz 1 \
            --drop_last True \
            --add_gan True \
            --gan_dis_type ${gan_dis_type}\
            --gan_dis_loss ${gan_dis_loss}\
            --lambda_gan ${lambda_gan} \
            ${@:1}
        done
    done
done
