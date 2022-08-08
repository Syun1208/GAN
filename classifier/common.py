import torch.nn as nn
from PIL import Image
from pytorchcv.model_provider import get_model
from torch.utils.data import Dataset

image_directory_dict = {
    "baseline": {
        "train": ["data/train"],
        "val": "data/val",
        "test": "data/test"
    },
    "esrgan": {
        "train": ["data/esrgan/train"],
        "val": "data/esrgan/val",
        "test": "data/esrgan/test"
    },
    "wgan_gp": {
        "train": ["data/wgan_gp/images/", "data/train"],
        "val": "data/val",
        "test": "data/test"
    },
    "esrgan_wgan_gp": {
        "train": ["data/esrgan_wgan_gp/", "data/esrgan/train"],
        "val": "data/esrgan/val",
        "test": "data/esrgan/test"
    },
    "esrgan_wgan_gp_fd": {
        "train": ["data/esrgan_wgan_gp_fd/", "data/esrgan/train"],
        "val": "data/esrgan/val",
        "test": "data/esrgan/test"
    },
}

plot_title_dict = {
    "vgg19": "VGG19",
    "resnet101": "ResNet101",
    "efficientnet_b2b": "EfficientNet",
    "baseline": "Baseline",
    "esrganwgan": "ESRGAN + WGAN-GP",
    "esrgan": "ESRGAN",
    "esrgan_wgan_gp_fd": "ESRGAN + WGAN-GP + Face Detection"
}


class EmotionDataset(Dataset):
    def __init__(self, img_list, label_list, transform=None):
        self.image = img_list
        self.label = label_list
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.image[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.label[idx]

        return img, label

    def __len__(self):
        return len(self.image)


class CNNModel(nn.Module):
    def __init__(self, model_name):
        super(CNNModel, self).__init__()
        # Load pretrained network as backbone
        pretrained = get_model(model_name, pretrained=True)
        # remove last layer of fc
        self.backbone = pretrained.features
 #       pretrained.output.fc3 = nn.Linear(4096, 7)
        self.output = pretrained.output
        self.classifier = nn.Linear(1000, 7)

#        nn.init.zeros_(self.classifier.fc3.bias)
#        nn.init.normal_(self.classifier.fc3.weight, mean=0.0, std=0.02)

        del pretrained

    def forward(self, x):
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1)
        x = self.output(x)
        x = self.classifier(x)

        return x

    def freeze_backbone(self):
        """Freeze the backbone network weight"""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Freeze the backbone network weight"""
        for p in self.backbone.parameters():
            p.requires_grad = True
