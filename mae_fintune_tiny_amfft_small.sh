CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1003 main_finetune.py \
--accum_iter 4 \
--batch_size 64 \
--model vit_small_patch16 \
--epochs 100 \
--data 'tinyimagenet' \
--data_path '/mnt/workspace/data/tiny-imagenet-200' \
--nb_classes 200 \
--blr 1.0e-3 --layer_decay 0.75 --cls_token \
--weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 \
--dist_eval \
--output_dir './amonly_small_preontiny_400_finetune_tinyimagenet_1' --log_dir './amonly_small_preontiny_400_finetune_tinyimagenet_1' \
--finetune '/mnt/workspace/AMT/mae-AMT/amonly_small_400_pretrain_tiny/checkpoint-399.pth' 
