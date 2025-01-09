CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1000 main_finetune.py \
--accum_iter 4 \
--batch_size 64 \
--model vit_base_patch16 \
--epochs 100 \
--data 'cifar10' \
--data_path '/earth-nas/datasets' \
--nb_classes 10 \
--blr 1.0e-3 --layer_decay 0.75 --cls_token \
--weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 \
--dist_eval \
--output_dir './maebase_400_finetune_cifar10_4' --log_dir './maebase_400_finetune_cifar10_4' \
--finetune '/mnt/workspace/baseline/mae/mae_base_400/checkpoint-399.pth' 
