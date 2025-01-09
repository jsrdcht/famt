# CUDA_VISIBLE_DEVICES=7 \
python -m torch.distributed.launch --master_port=1008 --nproc_per_node=8 main_linprobe_cifar100.py \
--accum_iter 4 \
--batch_size 512 \
--model vit_small_patch16 --cls_token \
--epochs 90 \
--blr 0.1 \
--weight_decay 0.0 \
--dist_eval \
--data_path '/mnt/workspace/data' \
--output_dir './amtfft_small_800_linprobe_cifar100_070' --log_dir './amtfft_small_800_linprobe_cifar100_070' \
--finetune '/mnt/workspace/AMT/mae-AMT/amtfft_small_800_pretrain_80/checkpoint-799.pth'