import argparse
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from common import CNNModel, EmotionDataset, image_directory_dict

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
weight_list = [os.path.join(weight_dir, path)
               for path in os.listdir(weight_dir)]


test_dir = image_directory_dict[opt.dataset]["test"]
test_img_list = list()
test_label_list = list()
for class_number in range(7):
    new_images = glob.glob(os.path.join(test_dir, str(class_number), '*.png'))
    test_img_list += new_images
    for _ in range(len(new_images)):
        test_label_list.append(class_number)


def accuracy(prediction, ground_truth):
    num_correct = (np.array(prediction) == np.array(ground_truth)).sum()
    return num_correct / len(prediction)


img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_ds = EmotionDataset(
    test_img_list, test_label_list, transform=img_transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

BATCH_SIZE = 64
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE,
                     num_workers=4, pin_memory=True)

weight_list = [os.path.join(weight_dir, path)
               for path in os.listdir(weight_dir)]

model = CNNModel(opt.classifier)
#model = nn.DataParallel(model)
model = model.to(device)
max_acc = 0
max_weight = ""

for weight in weight_list:
    if torch.cuda.is_available():
        state_dict = torch.load(weight)
    else:
        state_dict = torch.load(weight, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    model.eval()
    predictions = []
    ground_truths = []

    for img, label in tqdm(test_dl):
        img = img.to(device)
        with torch.no_grad():
            logits = model(img)
            prediction = torch.argmax(logits, dim=1)

            predictions.extend(prediction.tolist())
            ground_truths.extend(label.tolist())

    acc = accuracy(predictions, ground_truths)
    if acc > max_acc:
        max_acc = acc
        max_weight = weight

    print('test accuracy = {}, weight path = {}'.format(acc, weight))

print('highest test accuracy = {}, weight path = {}'.format(max_acc, max_weight))
