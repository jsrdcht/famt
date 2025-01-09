CUDA_VISIBLE_DEVICES=6 \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1006 main_finetune.py \
--accum_iter 16 \
--batch_size 64 \
--model vit_small_patch16 \
--epochs 100 \
--data 'cifar100' \
--data_path '/mnt/workspace/data' \
--nb_classes 100 \
--blr 1.0e-3 --layer_decay 0.75 --cls_token \
--weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 \
--dist_eval \
--output_dir './maebase_small_800_finetune_cifar100_1' --log_dir './maebase_small_800_finetune_cifar100_1' \
--finetune '/mnt/workspace/baseline/mae/mae_small_800_16/checkpoint-799.pth' 
