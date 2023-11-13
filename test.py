from model.vgg16_unet import VGG16Unet
from torch.utils.tensorboard import SummaryWriter
import torch
model = VGG16Unet()
print(model)

image = torch.zeros((1, 3, 256, 256))
print(image.shape)

model.eval()
result = model(image)
print(result.shape)