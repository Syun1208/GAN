import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128,
                    help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100,
                    help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=255,
                    help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3,
                    help="number of image channels")
parser.add_argument("--quality", type=str, default="baseline",
                    choices=["baseline", "esrgan"],
                    help="type of image to generate")
parser.add_argument("--face_detection", action="store_true",
                    help="use face detection")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

weight_file_dict = {
    "esrgan": "weights/esrgan_wgan_gp",
    "baseline": "weights/wgan_gp"
}

output_dir_dict = {
    "esrgan": "data/esrgan_wgan_gp",
    "baseline": "data/wgan_gp"
}

weight_file_dir = weight_file_dict[opt.quality]


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


def face_detector(img_path):
    # Load the cascade
    cascPath = "weights/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)
    # Read image
    img = cv2.imread(img_path)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    return len(faces)


generator = Generator()

if cuda:
    # generator = nn.DataParallel(generator)
    generator = generator.cuda()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

for class_number in range(7):
    state_dict = torch.load(os.path.join(
        weight_file_dir, str(class_number), "weight.pth"))
    generator.load_state_dict(state_dict)

    root_dir = os.path.join(output_dir_dict[opt.quality], str(class_number))
    os.makedirs(root_dir, exist_ok=True)
    img_list = os.listdir(
        root_dir) + os.listdir(os.path.join("data/baseline/train", str(class_number)))
    img_list_len = len(img_list)  # Total number of generated + original images

    while img_list_len < 10000:
        z = Variable(Tensor(np.random.normal(
            0, 1, (opt.batch_size, opt.latent_dim))))
        fake_imgs = generator(z)

        image_path = os.path.join(root_dir, "{img_list_len}_generated.png")
        save_image(fake_imgs.data[0], image_path, normalize=True)

        if opt.face_detector:
            if face_detector(image_path) == 0:
                os.remove(image_path)
            else:
                img_list_len += 1
        else:
            img_list_len += 1

        print(img_list_len)
