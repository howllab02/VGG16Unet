import torch
from tqdm import tqdm
from fn.metrics import dice_coe


class Trainer:
    def __init__(self, model, device, optimizer, loss_fn):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def fit(self, max_epoch, training_data):
        self.model = self.model.to(self.device)
        for epoch in range(max_epoch):
            print(f"Epoch [{epoch}/{max_epoch}]:")
            total_dice = 0
            total_loss = 0
            for data in tqdm(training_data):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                dice = dice_coe(outputs, labels)
                total_dice += dice
                total_loss += loss
            mean_dice_metrics = total_dice/len(training_data)
            mean_loss = total_loss/len(training_data)
            print(f"\n--- mean dice:{mean_dice_metrics}, dice loss:{mean_loss}")


