CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1004 main_finetune.py \
--accum_iter 4 \
--batch_size 64 \
--model vit_base_patch16 \
--epochs 100 \
--data 'cifar100' \
--data_path '/mnt/workspace/data' \
--nb_classes 100 \
--blr 1.0e-3 --layer_decay 0.75 --cls_token \
--weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 \
--dist_eval \
--output_dir './maebase_400_finetune_cifar100_4' --log_dir './maebase_400_finetune_cifar100_4' \
--finetune '/mnt/workspace/baseline/mae/mae_base_400/checkpoint-399.pth' 
