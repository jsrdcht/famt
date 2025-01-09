CUDA_VISIBLE_DEVICES=7 \
python -m torch.distributed.launch --master_port=1008 --nproc_per_node=1 main_linprobe_cifar100.py \
--accum_iter 32 \
--batch_size 512 \
--model vit_small_patch16 --cls_token \
--epochs 90 \
--blr 0.1 \
--weight_decay 0.0 \
--dist_eval \
--data_path '/mnt/workspace/data' \
--output_dir './amt6015_800_80_linprobe_cifar100_1' --log_dir './amt6015_800_80_linprobe_cifar100_1' \
--finetune '/mnt/workspace/AMT/mae-AMT/6015relast_800_80/checkpoint-799.pth'