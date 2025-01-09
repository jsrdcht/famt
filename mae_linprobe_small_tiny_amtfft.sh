CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --master_port=1000 --nproc_per_node=1 main_linprobe_IN200.py \
--accum_iter 32 \
--batch_size 512 \
--model vit_small_patch16 --cls_token \
--epochs 90 \
--blr 0.1 \
--weight_decay 0.0 \
--dist_eval \
--data_path '/mnt/workspace/data/tiny-imagenet-200' \
--output_dir './amtonly_small_preontiny_400_linprobe_tiny_1' --log_dir './amtonly_small_preontiny_400_linprobe_tiny_1' \
--finetune '/mnt/workspace/AMT/mae-AMT/amtonly_small_400_pretrain_tiny/checkpoint-399.pth'