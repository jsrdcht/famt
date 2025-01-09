python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
--data_path '/mnt/workspace/data/tiny-imagenet-200' \
--output_dir './amtonly_small_400_pretrain_tiny' \
--log_dir './amtonly_small_400_pretrain_tiny' \
--batch_size 128 \
--accum_iter 4 \
--blr 1.5e-4 \
--epochs 400 \
--resume '/mnt/workspace/baseline/mae/mae_small_preontiny_400/checkpoint-40.pth'