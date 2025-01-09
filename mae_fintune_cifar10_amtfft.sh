CUDA_VISIBLE_DEVICES=1 \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1001 main_finetune.py \
--accum_iter 16 \
--batch_size 64 \
--model vit_small_patch16 \
--epochs 100 \
--data 'cifar10' \
--data_path '/earth-nas/datasets' \
--nb_classes 10 \
--blr 1.0e-3 --layer_decay 0.75 --cls_token \
--weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 \
--dist_eval \
--output_dir './amtfft_800_80_finetune_cifar10_1' --log_dir './amtfft_800_80_finetune_cifar10_1' \
--finetune '/mnt/workspace/AMT/mae-AMT/amtfft_small_800_pretrain_80/checkpoint-799.pth' 