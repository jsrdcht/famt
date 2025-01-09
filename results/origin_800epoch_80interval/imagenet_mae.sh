CUDA_VISIBLE_DEVICES=0,1,3,6 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29509 main_pretrain.py \
    --data_path /workspace/sync/imagenet-1k \
    --output_dir /workspace/sync/famt/results/origin_800epoch_80interval \
    --log_dir /workspace/sync/famt/results/origin_800epoch_80interval \
    --batch_size 128 \
    --accum_iter 8 \
    --blr 1.5e-4 \
    --epochs 800 \
    --loc_interval 80