from model.vgg16_unet import VGG16Unet
from torch.utils.tensorboard import SummaryWriter
import torch
model = VGG16Unet()
print(model)

example_input = torch.zeros((3, 256, 256)).unsqueeze(0)

yhat = model(example_input)
print(yhat)

model.eval()
writer = SummaryWriter('runs/unet_model')
writer.add_graph(model, example_input)
writer.close()

