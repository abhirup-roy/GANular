import torch

# Training parameters
epochs = 500
batch_size = 32
learning_rate = 0.0002

noise_size = 256
embed_size = 16

n_proj = 64

if torch.cuda.is_available():
    torch.device(device="cuda")
elif torch.backends.mps.is_available():
    torch.device(device="mps")
else:
    torch.device(device="cpu")

data_dir = "/Volumes/AbhiSSD_SamsungT7/ImgParticl8/cropped_particles"
