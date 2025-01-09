OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_linprobe.py \
    --accum_iter 4 \
    --batch_size 512 \
    --model vit_base_patch16 --cls_token \
    --finetune '/mnt/workspace/AMT/mae-AMT/6015relast_800_80/checkpoint-799.pth' \
    --epochs 90 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval --data_path '/earth-nas/datasets/imagenet-1k' \
    --output_dir './linprobe_imagenet_amt6015_800_80' \
    --log_dir './linprobe_imagenet_attn-drivenmask_800_80'
# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
#     --accum_iter 4 \
#     --batch_size 32 \
#     --model vit_base_patch16 \
#     --finetune '/mnt/workspace/AMT/mae-AMT/6015relast1/checkpoint-399.pth' \
#     --epochs 100 \
#     --blr 1e-3 --layer_decay 0.75 \
#     --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
#     --dist_eval --data_path '/earth-nas/datasets/imagenet-1k' \
#     --output_dir './finetune_imagenet' \
#     --log_dir './finetune_imagenet'