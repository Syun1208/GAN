import os

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import PIL

dataroot = "data/baseline/train"

total_list = []
label_list = []
for i in range(7):
    emotion_path = os.path.join(dataroot, i)
    emotion_files = os.listdir(emotion_path)
    total_list += [os.path.join(emotion_path, path) for path in emotion_files]
    label_list += [0 for i in range(len(emotion_files))]

df = pd.DataFrame({'Image': total_list})
df['Label'] = label_list

ia.seed(1)

# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8

seq1 = iaa.AdditiveGaussianNoise(scale=(0, 11))

seq2 = iaa.AdditiveGaussianNoise(scale=(0, 31))

seq3 = iaa.AdditiveGaussianNoise(scale=(0, 51))

seq4 = iaa.AdditiveGaussianNoise(scale=(0, 71))

seq5 = iaa.AdditiveLaplaceNoise(scale=(0, 51))

seq6 = iaa.AdditivePoissonNoise(51)

seq7 = iaa.Sequential([
    iaa.OneOf([iaa.AdditivePoissonNoise(51),
               iaa.AdditiveLaplaceNoise(scale=(0, 51)),
               iaa.AdditiveGaussianNoise(scale=(0, 51))
               ])])

for i in range(len(df['Image'])):

    images = np.array(PIL.Image.open(df['Image'][i]).convert('RGB'))

    if df['Label'][i] == 1:
        l = 24

    elif df['Label'][i] == 0 or df['Label'][i] == 2 or df['Label'][i] == 5:
        l = 2

    elif df['Label'][i] == 3 or df['Label'][i] == 4 or df['Label'][i] == 6:
        l = 1

    for k in range(l):
        aug_img1 = PIL.Image.fromarray(seq1(images=images))
        aug_img1 = aug_img1.save('data/gaussian_noise_v0/' +
                                 df['Label'][i] + '/' + 'augmented_image_seq0_' + str(i) + '_' + str(k) + '.png')

        aug_img2 = PIL.Image.fromarray(seq2(images=images))
        aug_img2 = aug_img2.save('data/gaussian_noise_v1/' +
                                 df['Label'][i] + '/' + 'augmented_image_seq0_' + str(i) + '_' + str(k) + '.png')

        aug_img3 = PIL.Image.fromarray(seq3(images=images))
        aug_img3 = aug_img3.save('data/gaussian_noise_v2/' +
                                 df['Label'][i] + '/' + 'augmented_image_seq2_' + str(i) + '_' + str(k) + '.png')

        aug_img4 = PIL.Image.fromarray(seq4(images=images))
        aug_img4 = aug_img4.save('data/gaussian_noise_v3/' +
                                 df['Label'][i] + '/' + 'augmented_image_seq3_' + str(i) + '_' + str(k) + '.png')

        aug_img5 = PIL.Image.fromarray(seq5(images=images))
        aug_img5 = aug_img5.save('data/laplacian_noise/' +
                                 df['Label'][i] + '/' + 'augmented_image_seq2_' + str(i) + '_' + str(k) + '.png')

        aug_img6 = PIL.Image.fromarray(seq6(images=images))
        aug_img6 = aug_img6.save('data/poisson_noise/' +
                                 df['Label'][i] + '/' + 'augmented_image_seq3_' + str(i) + '_' + str(k) + '.png')

        aug_img7 = PIL.Image.fromarray(seq7(images=images))
        aug_img7 = aug_img7.save('data/mixed_noise/' +
                                 df['Label'][i] + '/' + 'augmented_image_seq5_' + str(i) + '_' + str(k) + '.png')
