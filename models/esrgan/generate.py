import os.path as osp
import os
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

# RRDB_ESRGAN_x4.pth OR RRDB_PSNR_x4.pth
model_path = "weights/esrgan/RRDB_ESRGAN_x4.pth"
output_path = "data/esrgan/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_image(input_img_path, output_image_path):
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
        os.makedirs(output_image_path, exist_ok=True)
        cv2.imwrite(os.path.join(output_image_path,
                                 "{:s}_rlt.png".format(base)), output)


for class_number in range(7):
    for type in ["train", "val", "test"]:
        input_img_path = os.path.join("data/baseline", type, f"{class_number}/*")
        output_image_path = os.path.join(output_path, type, f"{class_number}")

        generate_image(input_img_path, output_image_path)
