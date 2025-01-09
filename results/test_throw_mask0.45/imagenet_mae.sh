OUTPUT_DIR=/workspace/sync/famt/results/test_throw
SCRIPT_PATH=$(realpath "$0")
cp $SCRIPT_PATH $OUTPUT_DIR
CUDA_VISIBLE_DEVICES=0,1,5,7 torchrun --nproc_per_node=4 --master_port=29501 main_pretrain.py --data_path /workspace/sync/imagenet-1k --output_dir $OUTPUT_DIR --log_dir $OUTPUT_DIR --batch_size 128 --accum_iter 4 --blr 1.5e-4 --epochs 200 --loc_interval 20 --resume $OUTPUT_DIR/checkpoint-40.pth --throw_ratio 0.0