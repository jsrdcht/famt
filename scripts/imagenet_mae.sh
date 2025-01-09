OUTPUT_DIR=/workspace/sync/famt/results/remove_fft
mkdir -p $OUTPUT_DIR
SCRIPT_PATH=$(realpath "$0")
cp $SCRIPT_PATH $OUTPUT_DIR
CUDA_VISIBLE_DEVICES=0,1,3,5 torchrun --nproc_per_node=4 --master_port=29509 main_pretrain.py --data_path /workspace/sync/imagenet-1k --output_dir $OUTPUT_DIR --log_dir $OUTPUT_DIR --batch_size 128 --accum_iter 8 --blr 1.5e-4 --epochs 200 --loc_interval 40