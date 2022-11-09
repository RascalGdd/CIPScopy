from generator import CIPSskip as Generator
from noise import mixing_noise

device = "cuda"

noise = mixing_noise(1, 512, 0, device)

generator = Generator(size=256, hidden_size=512, style_dim=512, n_mlp=8,
                      activation=None, channel_multiplier=2,
                      ).to(device)
print(noise)