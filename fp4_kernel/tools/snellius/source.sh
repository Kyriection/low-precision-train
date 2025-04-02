#!/bin/bash

module load 2024
module load Anaconda3/2024.06-1
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh" ]; then
        . "/sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1/etc/profile.d/conda.sh"
    else
        export PATH="/sw/arch/RHEL9/EB_production/2024/software/Anaconda3/2024.06-1/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate fp4