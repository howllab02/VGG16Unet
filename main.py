from fn.losses import DiceLoss
from dataset import BrainTumor, transforms
from train import Trainer
from model.vgg16_unet import VGG16Unet
from torch.utils.data import DataLoader, Dataset
import os
import torch

dataset_folder = "dataset/images"
files = os.listdir(dataset_folder)
list_path = []
for file in files:
    file_path = os.path.join(dataset_folder, file)
    list_path.append(file_path)
files = list_path

data = BrainTumor(path_data=files, transform=transforms)
print(len(data))
loader = DataLoader(data, num_workers=2, batch_size=256)
images, labels = next(iter(loader))
print(len(images))
print(len(labels))

model = VGG16Unet()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = DiceLoss()

trainer = Trainer(model, "cuda", optimizer, loss)
trainer.fit(10, training_data=loader)