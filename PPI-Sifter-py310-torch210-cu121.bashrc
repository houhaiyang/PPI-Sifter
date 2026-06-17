# Source default setting
[ -f /etc/bashrc ] && . /etc/bashrc

# 初始化 module 环境（优先于其他环境配置）
source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
module purge

# 加载必要的工具链和库（通过 module 管理）
module load compilers/cuda/12.1.0
module load compilers/gcc/10.5.0
module load libs/cudnn/8.9.7_cuda12
module load libs/openblas/0.3.20_gcc10.5.0
module load tools/cmake/3.27.0
module load tools/ccache/4.10.2


# User environment PATH（基础路径配置，不与 module 冲突）
PATH="$HOME/.local/bin:$HOME/bin:$PATH"
export PATH

# 命令行前缀（豪华三行样式，保留原个性化设置）
export PS1="\e[1;2;34m█ █\e[0m\e[32m╭─\e[32m\u@\e[32m\$(printf '%-20.20s' \"\h\")\e[0m \e[91;1m︹︹ \e[0m\e[32mUCAS─AI ⇆ BGI─Beijing──<<<\n\e[0m\e[1;2;34m███\e[0m\e[32m│\e[32m time: \$(date +%Y-%m-%d) \$(date +%H:%M:%S)\e[34m noBUG \e[91;1m︺︺\e[0m\e[32m path:『\e[33m\w\e[32m』\n\e[1;2;34m█ █\e[0m\e[32m╰─\e[1;93m>> \e[32;1m\e[0m"

# 快捷键别名（保留原设置）
alias ts='tree -C -L 1'
alias ks='tree -C -L 1'
alias kss='tree -C -L 2'
alias s1='du -lh --max-depth=1'
alias s2='du -lh --max-depth=2'
alias sshx='ssh cyclone001-agent-64'
alias ssh64='ssh cyclone001-agent-64'
alias ssh56='ssh cyclone001-agent-56'

# nccl
export LD_LIBRARY_PATH="/home/HPCBase/libs/nccl/2.19.3-1+cuda12.2/lib:$LD_LIBRARY_PATH"

# 确保在导入其他可能使用OpenMP的库（如PyTorch、TensorFlow）之前，先导入scikit-learn
# export LD_PRELOAD=/home/share/huadjyin/home/houhaiyang/.conda/envs/ppisifter/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-947d5fa1.so.1.0.0

# ppisifter 环境与 HF_HOME 配置（保留原设置，与 module 无冲突）
export PATH="/home/share/huadjyin/home/houhaiyang/.conda/envs/ppisifter/bin:$PATH"


