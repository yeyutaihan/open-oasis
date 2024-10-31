# Oasis 500M

![](./media/arch.png)

![](./media/thumb.png)

Oasis is an interactive world model developed by [Decart](https://www.decart.ai/) and [Etched](https://www.etched.com/). Based on diffusion transformers, Oasis takes in user keyboard input and generates gameplay in an autoregressive manner. Here we release the weights for Oasis 500M, a downscaled version of the model, along with inference code for action-conditional frame generation.

For more details, see our [joint blog post](https://oasis-model.github.io/) to learn more.

## Setup

```
git lfs install
git clone https://huggingface.co/Etched/oasis-500m
cd oasis-500m
pip install -r requirements.txt
```

## Basic Usage
We include a basic inference script that loads a prompt frame from a video and generates additional frames conditioned on actions.
```
cd oasis-500m
python inference.py
```
The resulting video will be saved to `video.mp4`.

