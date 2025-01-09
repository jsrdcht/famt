CUDA_VISIBLE_DEVICES=0,3,5,6 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29510 main_pretrain.py \
    --data_path /workspace/sync/imagenet-1k \
    --output_dir /workspace/sync/famt/results/maebase_removethrowmiddletokens \
    --log_dir /workspace/sync/famt/results/maebase_removethrowmiddletokens \
    --batch_size 64 \
    --model mae_vit_base_patch16 \
    --accum_iter 16 \
    --blr 1.5e-4 \
    --epochs 200 \
    --loc_interval 40