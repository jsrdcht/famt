CUDA_VISIBLE_DEVICES=2 \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1002 main_finetune.py \
--accum_iter 16 \
--batch_size 64 \
--model vit_base_patch16 \
--epochs 100 \
--data 'cifar10' \
--data_path '/earth-nas/datasets' \
--nb_classes 10 \
--blr 1.0e-3 --layer_decay 0.75 --cls_token \
--weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 \
--dist_eval \
--output_dir './attnmask_800_160_finetune_cifar10_1' --log_dir './attnmask_800_160_finetune_cifar10_1' \
--finetune '/mnt/workspace/AMT/mae-AMT/attn-drivenmask_800_160/checkpoint-799.pth' 
