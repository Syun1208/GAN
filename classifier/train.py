import argparse
import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from common import (CNNModel, EmotionDataset, image_directory_dict,
                    plot_title_dict)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="baseline",
                    choices=["baseline", "esrgan", "wgan_gp", 
                             "esrgan_wgan_gp", "esrgan_wgan_gp_fd"],
                    help="augmented dataset to use")
parser.add_argument("--classifier", type=str, default="vgg19",
                    choices=["vgg19", "resnet101", "efficientnet-b2b"],
                    help="classifier type")
opt = parser.parse_args()

weight_dir = "weights/{}/{}/".format(opt.classifier, opt.dataset)
graph_dir = "classifier/graphs/{}/{}/".format(opt.classifier, opt.dataset)
os.makedirs(weight_dir, exist_ok=True)
os.makedirs(graph_dir, exist_ok=True)

train_dirs = image_directory_dict[opt.dataset]["train"]
train_img_list = list()
train_label_list = list()
for train_dir in train_dirs:
    for class_number in range(7):
        new_images = glob.glob(os.path.join(train_dir, str(class_number), '*.png'))
        train_img_list.extend(new_images)
        for _ in range(len(new_images)):
            train_label_list.append(class_number)

valid_dir = image_directory_dict[opt.dataset]["val"]
valid_img_list = list()
valid_label_list = list()
for class_number in range(7):
    new_images = glob.glob(os.path.join(valid_dir, str(class_number), '*.png'))
    valid_img_list.extend(new_images)
    for _ in range(len(new_images)):
        valid_label_list.append(class_number)


def accuracy(prediction, ground_truth):
    num_correct = (np.array(prediction) == np.array(ground_truth)).sum()
    return num_correct / len(prediction)


img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_ds = EmotionDataset(
    train_img_list, train_label_list, transform=img_transform)
valid_ds = EmotionDataset(
    valid_img_list, valid_label_list, transform=img_transform)


def train_model():
    running_loss = []
    running_corrects = []
    model.train()

    for img, label in tqdm(train_dl):
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        logits = model(img)
        loss = criterion(logits, label)
        _, preds = torch.max(logits, 1)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item() * img.size(0))
        running_corrects.append(torch.sum(preds == label.data))

    epoch_loss = sum(running_loss)/len(running_loss)
    epoch_acc = float(sum(running_corrects))/len(running_corrects)

    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)

    model.eval()

    predictions = []
    ground_truths = []

    for img, label in tqdm(valid_dl):
        img = img.to(device)
        with torch.no_grad():
            logits = model(img)
            prediction = torch.argmax(logits, dim=1)

            predictions.extend(prediction.tolist())
            ground_truths.extend(label.tolist())

    acc = accuracy(predictions, ground_truths)
    val_acc.append(acc)

    with open(os.path.join(graph_dir, f"{opt.classifier}_{opt.dataset}_log.txt"), "w") as f:
        f.write("Training Loss: \n")
        f.writelines("%s\n" % loss for loss in train_loss)
        f.write("\n")
        f.write("Training Accuracy: \n")
        f.writelines("%s\n" % acc for acc in train_acc)
        f.write("\n")
        f.write("Validation Accuracy: \n")
        f.writelines("%s\n" % acc for acc in val_acc)

    return acc


highest_acc = 0
train_loss = []
train_acc = []
val_acc = []

EPOCHS = 2
BATCH_SIZE = 64
LR = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                      shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE,
                      num_workers=4, pin_memory=True)

model = model = CNNModel(opt.classifier).to(device)
model.freeze_backbone()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, len(train_dl), T_mult=EPOCHS*len(train_dl))
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    train_model()

EPOCHS = 100
BATCH_SIZE = 64
LR = 25e-6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model = CNNModel(opt.classifier)
model.unfreeze_backbone()
#model = nn.DataParallel(model)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-7)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, len(train_dl), T_mult=len(train_dl)*EPOCHS)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    acc = train_model()

    if acc > highest_acc:
        highest_acc = acc
        torch.save(model.state_dict(), os.path.join(
            weight_dir, "weights_epoch_{}_acc_{}.pth".format(epoch, acc)))


# Graph plotting
val_acc = [x * 100 for x in val_acc]

mpl.use('Agg')

plt.ioff()

fig, ax1 = plt.subplots()

ax1.set_xlabel('Epochs')
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

ax1.set_ylabel('Loss')
ax1.plot(train_loss, color='tab:blue')
ax1.tick_params(axis='y')
ax1.legend(["Training Loss"])

ax2 = ax1.twinx()

ax2.set_ylabel('Accuracy (%)')
ax2.plot(train_acc, color='tab:orange')
ax2.plot(val_acc, color='tab:red')
ax2.tick_params(axis='y')
ax2.legend(["Training Accuracy", "Validation Accuracy"])

plt.title(
    f'{plot_title_dict[opt.classifier]} ({plot_title_dict[opt.dataset]})')

plt.savefig(os.path.join(
    graph_dir, f'{opt.classifier}_{opt.dataset}_loss_acc.png'), bbox_inches='tight')
