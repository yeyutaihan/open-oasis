#!/bin/bash
#SBATCH --job-name=oasis         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:15:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=2200012939@stu.pku.edu.cn
module purge
module load anaconda3/2024.6
conda activate /home/tc0786/anaconda3/envs/oasis
python /home/tc0786/Project/open-oasis/generate.py \
    --oasis-ckpt /home/tc0786/Project/open-oasis/data/models/oasis500m/oasis500m.safetensors \
    --vae-ckpt /home/tc0786/Project/open-oasis/data/models/vit/vit-l-20.safetensors \
    --output-path /home/tc0786/Project/open-oasis/data/outputs/oasis/test/all_forward.mp4 \
    --prompt-path /home/tc0786/Project/open-oasis/sample_data/sample_image_0.png \
    --actions-path /home/tc0786/Project/open-oasis/sample_data/all_forward.one_hot_actions.pt \
    --num-frames 960