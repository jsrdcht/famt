CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.launch --master_port=1006 --nproc_per_node=4 main_linprobe_cifar100.py \
--accum_iter 8 \
--batch_size 512 \
--model vit_base_patch16 --cls_token \
--epochs 90 \
--blr 0.1 \
--weight_decay 0.0 \
--dist_eval \
--data_path '/mnt/workspace/data' \
--output_dir './maebase_400_linprobe_cifar100_4' --log_dir './maebase_400_linprobe_cifar100_4' \
--finetune '/mnt/workspace/baseline/mae/mae_base_400/checkpoint-399.pth'