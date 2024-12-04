import os

# os.system("huggingface-cli download --resume-download --local-dir-use-symlinks False Etched/oasis-500m oasis500m.safetensors --local-dir /home/tc0786/Project/open-oasis/data/models/oasis500m")
os.system("huggingface-cli download --resume-download --local-dir-use-symlinks False Etched/oasis-500m vit-l-20.safetensors --local-dir /home/tc0786/Project/open-oasis/data/models/vit")
