CUDA_VISIBLE_DEVICES=6 \
python -m torch.distributed.launch --master_port=1017 --nproc_per_node=1 main_linprobe_cifar10.py \
--accum_iter 32 \
--batch_size 512 \
--model vit_small_patch16 --cls_token \
--epochs 90 \
--blr 0.1 \
--weight_decay 0.0 \
--dist_eval \
--data_path '/earth-nas/datasets' \
--output_dir './amfft_small_800_linprobe_cifar10_1' --log_dir './amfft_small_800_linprobe_cifar10_1' \
--finetune '/mnt/workspace/AMT/mae-AMT/am_small_800_pretrain_80_fft/checkpoint-799.pth'
