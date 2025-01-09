CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 --master_port=1001 main_finetune.py \
--accum_iter 4 \
--batch_size 128 \
--model vit_small_patch16 \
--epochs 100 \
--data 'tinyimagenet' \
--data_path '/mnt/workspace/data/tiny-imagenet-200' \
--nb_classes 200 \
--blr 1.0e-3 --layer_decay 0.75 --cls_token \
--weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 \
--dist_eval \
--output_dir './amtonly_preontiny_400_finetune_tinyimagenet_1' --log_dir './amtonly_preontiny_400_finetune_tinyimagenet_1' \
--finetune '/workspace/sync/famt/results/origin_200epoch_20interval/checkpoint-199.pth' 
