from torch.utils.data import DataLoader, Dataset
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
import os

transforms = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
                                 transforms.Resize((256, 256), antialias=True),
                                 transforms.RandomVerticalFlip(p=0.5),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
                                 transforms.Normalize((0.5,), (0.5,))])


class BrainTumor(Dataset):
    def __init__(self, path_data, transform):
        self.data = path_data
        self.transform = transform
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        path_img = self.data[index]

        path_f = path_img.split(".")[0]
        image = Image.open(path_f + ".tif").convert("RGB")
        label = Image.open(path_f.replace("images", "labels") + "_mask.tif").convert("L")
        image, label = self.transform(image, label)
        return image, label


if __name__ == "__main__":
    dataset_folder = "dataset/images"
    files = os.listdir(dataset_folder)
    list_path = []
    for file in files:
        file_path = os.path.join(dataset_folder, file)
        list_path.append(file_path)
    files = list_path

    data = BrainTumor(path_data=files, transform=transforms)
    print(len(data))
    loader = DataLoader(data, num_workers=2, batch_size=32)
    images, labels = next(iter(loader))
    print(len(images))
    print(len(labels))
