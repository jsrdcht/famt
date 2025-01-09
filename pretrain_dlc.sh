python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
--data_path '/earth-nas/datasets/imagenet-1k' \
--output_dir './6015relast_1600' \
--log_dir './6015relast_1600' \
--batch_size 128 \
--accum_iter 4 \
--blr 1.5e-4 \