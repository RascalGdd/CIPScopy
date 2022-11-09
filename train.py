from generator import CIPSskip as Generator
from noise import mixing_noise
import torch
import tensor_transforms as tt

batch_size = 1
device = "cpu"

noise = mixing_noise(batch_size, 512, 0, device)

generator = Generator(size=256, hidden_size=512, style_dim=512, n_mlp=8,
                      activation=None, channel_multiplier=2,
                      ).to(device)

coords = tt.convert_to_coord_format(batch_size, 256, 512, integer_values=False)
img = torch.randn([batch_size, 3, 256, 512])
real_stack = torch.cat([img, coords], 1).to(device)
real_img, converted = real_stack[:, :3], real_stack[:, 3:]
fake_img, _ = generator(converted, noise)
print(fake_img.shape)

