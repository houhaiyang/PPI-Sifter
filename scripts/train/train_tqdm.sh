#!/bin/bash
#DSUB -n PPI-Sifter
#DSUB -N 1
#DSUB -A root.project.P24Z28400N0259_tmp2
#DSUB -q root.default
#DSUB -d "PPI-Sifter-Train"
#DSUB -T 180000
#DSUB -pn "cyclone001-agent-151"
#DSUB -R "cpu=32;gpu=4;mem=160000"
#DSUB -oo /home/share/huadjyin/home/houhaiyang/project/PPI-Sifter/outputs/logs/cyclone001-agent-151.out
#DSUB -eo /home/share/huadjyin/home/houhaiyang/project/PPI-Sifter/outputs/logs/cyclone001-agent-151.err

# 加载系统 Conda
source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh
# 加载 环境
source ~/bashrc/PPI-Sifter-py310-torch210-cu121.bashrc

# 转到工作路径
cd /home/share/huadjyin/home/houhaiyang/project/PPI-Sifter

nvidia-smi

echo "Starting PPI-Sifter Training！ $(date)"

CUDA_VISIBLE_DEVICES=2 python scripts/train/train_tqdm_log.py

echo "Completed at $(date)"

