#!/bin/bash
#BSUB -J 6015_rel_lincifar10_200
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q gpu_v100
#BSUB  -gpu "num=1:mode=exclusive_process:aff=yes"
# module load cuda-11.3
# module load gcc-8.2.0
# module load anaconda3
# source activate
# conda deactivate
# conda activate cu113
CUDA_VISIBLE_DEVICES=1 \
python -m torch.distributed.launch --master_port=1002 --nproc_per_node=1 main_linprobe_cifar10.py \
--accum_iter 32 \
--batch_size 512 \
--model vit_base_patch16 --cls_token \
--epochs 90 \
--blr 0.1 \
--weight_decay 0.0 \
--dist_eval \
--data_path '/earth-nas/datasets' \
--output_dir './maebase_small_800_linprobe_cifar10_1' --log_dir './maebase_small_800_linprobe_cifar10_1' \
--finetune '/mnt/workspace/AMT/mae-AMT/attn-drivenmask_800_80/checkpoint-799.pth'
