import os.path as osp
import os
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import argparse

# RRDB_ESRGAN_x4.pth OR RRDB_PSNR_x4.pth
model_path = "E:\\GAN\\weights\\esrgan\\RRDB_ESRGAN_x4.pth"
output_path = "E:\\GAN\\data\\esrgan"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img_dir_path', help='Image dir', default='E:\\GAN\\data\\baseline')
    parser.add_argument('--output_img_dir_path', type=str, default='E:\\GAN\\data\\esrgan')
    parser.add_argument('--width', help='image width', type=int, default=28)
    parser.add_argument('--height', help='image height', type=int, default=28)
    return parser.parse_args()


def generate_image(input_img_path, output_image_path, img_dim):
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    print("Model path {:s}. \nTesting...".format(model_path))

    idx = 0
    for path in glob.glob(input_img_path):
        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        print(idx, base)
        print('Successful !')
        # read images
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(
            img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze(
            ).float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        resized = cv2.resize(output, img_dim, interpolation=cv2.INTER_AREA)
        print('Resized Dimensions : ', resized.shape)
        if not os.path.exists(output_image_path):
            os.makedirs(output_image_path)
        cv2.imwrite(os.path.join(output_image_path,
                                 "{:s}_esrgan.jpg".format(base)), resized)


args = parse_arg()
dim = (args.width, args.height)
for class_number in range(10):
    for type in ["train", "val", "test"]:
        input_img_path = os.path.join(args.input_img_dir_path, type, f"{class_number}/*")
        output_image_path = os.path.join(args.output_img_dir_path, type, f"{class_number}")

        generate_image(input_img_path, output_image_path, dim)
