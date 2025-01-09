CUDA_VISIBLE_DEVICES=4 \
python -m torch.distributed.launch --master_port=1005 --nproc_per_node=1 main_linprobe_cifar100.py \
--accum_iter 32 \
--batch_size 512 \
--model vit_base_patch16 --cls_token \
--epochs 90 \
--blr 0.1 \
--weight_decay 0.0 \
--dist_eval \
--data_path '/mnt/workspace/data' \
--output_dir './attnmask_800_160_linprobe_cifar100_1' --log_dir './attnmask_800_160_linprobe_cifar100_1' \
--finetune '/mnt/workspace/AMT/mae-AMT/attn-drivenmask_800_160/checkpoint-799.pth'