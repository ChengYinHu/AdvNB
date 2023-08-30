import sys

import cv2
import numpy as np
import random
from PIL import  Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = models.resnet50(pretrained=True).eval().to(device)

# Neon patterns are generated in digital environments
def img_neon_effetc_digital(img, x, y, r, g, b, filenameSize, R):


    cv2.circle(img, (x, y), R, (r, g, b), -1)

    cv2.imwrite("result/55_" + filenameSize + ".jpg", img)

# Neon effect are generated in digital environments
def video_neon_effect_digital(img, cnt, I):

    if cnt == 0:
        return img

    height, width, n = img.shape

    mask = {
        1: cv2.imread("result/55_1.jpg"),
        2: cv2.imread("result/55_2.jpg"),
        3: cv2.imread("result/55_3.jpg"),
        4: cv2.imread("result/55_4.jpg"),
    }

    mask[cnt] = cv2.resize(mask[cnt], (width, height), interpolation=cv2.INTER_CUBIC)

    new_img = cv2.addWeighted(img, (1 - I), mask[cnt], I, 0)

    return new_img

#Neon patterns are generated in physical environments
def img_neon_effect_physical_adjust(R, img, x, y, filenameSize, neon1_x, neon1_Y, neon2_x, neon2_Y, neon3_x, neon3_y, neon4_x, neon4_y, r, g, b):

    cv2.circle(img, ( x + neon1_x, y + neon1_Y ), R, (b, 0, 0), -1)
    cv2.circle(img, ( x + neon2_x, y + neon2_Y ), R, (b, 0, 0), -1)
    cv2.circle(img, ( x + neon3_x, y + neon3_y ), R, (b, 0, 0), -1)
    cv2.circle(img, ( x + neon4_x, y + neon4_y ), R, (b, 0, 0), -1)
    cv2.imwrite("55_" + filenameSize + ".jpg", img)

def img_neon_effect_physical(R, img, filenameSize, neon1_x, neon1_Y, r, g, b):

    R = random.randint(45,50)
    i = random.randint(2, 5)
    if i == 1:
        cv2.circle(img, (neon1_x, neon1_Y), R, (0, 0, r), -1)

    if i == 2:
        cv2.circle(img, (neon1_x, neon1_Y), R, (0, g, 0), -1)

    if i == 3:
        cv2.circle(img, (neon1_x, neon1_Y), R, (b, 0, 0), -1)

    if i == 4:
        cv2.circle(img, (neon1_x, neon1_Y), R, (0, g, r), -1)

    if i == 5:
        cv2.circle(img, (neon1_x, neon1_Y), R, (b, 0, r), -1)

    cv2.imwrite("result/55_" + filenameSize + ".jpg", img)

# Neon effect are generated in physical environments
def video_neon_effect_physical(img, cnt, I):

    I = random.randint(3, 5)/10

    if cnt == 0:
        return img

    height, width, n = img.shape

    mask = {
        1: cv2.imread("result/55_1.jpg"),
        2: cv2.imread("result/55_2.jpg"),
        3: cv2.imread("result/55_3.jpg"),
        4: cv2.imread("result/55_4.jpg"),
    }

    mask[cnt] = cv2.resize(mask[cnt], (width, height), interpolation=cv2.INTER_CUBIC)

    new_img = cv2.addWeighted(img, (1 - I), mask[cnt], I, 0)

    return new_img

# Classification function
def classify(dir, net):

    img = Image.open(dir)

    img = img.convert("RGB")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,std = std)
    ])(img).to(device)

    f_image = net.forward(Variable(img[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()



    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:10]

    label = I[0]

    conf = f_image[label]

    return label, conf

# Constrained neon position function in physical environment
def random_neon_position_fuc():

    neon1_x = random(170, 200)
    neon1_y = random(120, 150)
    neon2_x = random(280, 330)
    neon2_y = random(120, 150)
    neon3_x = random(170, 200)
    neon3_y = random(230, 260)
    neon4_x = random(280, 330)
    neon4_y = random(230, 260)

    return neon1_x, neon1_y, neon2_x, neon2_y, neon3_x, neon3_y, neon4_x, neon4_y

# The function of adjusting the corlor
def adjust_corlor(datase_dir, img, I, R, r1, g1, b1, neon1_x, neon1_Y, neon2_x, neon2_Y, neon3_x, neon3_y, neon4_x, neon4_y):

    for r in range(r1 - 50, r1 + 1, 50):
        for g in range(g1 - 50, g1 + 1, 50):
            for b in range(b1 - 50, b1 + 1, 50):
                height, width, n = img.shape
                img_neon_effect_physical_adjust(R, img, 0, 0, "1", neon1_x, neon1_Y, neon2_x, neon2_Y, neon3_x, neon3_y, neon4_x, neon4_y, r, g, b)
                cap = cv2.VideoCapture(datase_dir)
                fps = cap.get(cv2.CAP_PROP_FPS)
                i1 = 1
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                videoWriter = cv2.VideoWriter('result_p1.jpg', fourcc, fps, (width, height))
                while (cap.isOpened()):
                    ret, frame = cap.read()
                    if ret:
                        frame = video_neon_effect_physical(frame, i1 % 5, I)
                        videoWriter.write(frame)
                        i1 += 1
                        c = cv2.waitKey(1)
                        if c == 27:
                            break
                    else:
                        break

                # img_check = Image.open('result_p1.jpg')
                # plt.imshow(img_check)
                # plt.show()
                label, confidence= classify('result_p1.jpg', net)
                label = int(label)
                print("label_adjust = ", label)

                if (label == 919) :
                    return 0

    return 1

# The function of adjusting the intensity
def adjust_intensity(datase_dir, img, I, R, r, g, b, neon1_x, neon1_Y, neon2_x, neon2_Y, neon3_x, neon3_y, neon4_x, neon4_y):

    I1 = adjust_corlor(datase_dir, img, I - 0.09, R, r, g, b, neon1_x, neon1_Y, neon2_x, neon2_Y, neon3_x, neon3_y, neon4_x, neon4_y)

    if I1 == 0 :
        return 0

    I2 = adjust_corlor(datase_dir, img, I, R, r, g, b, neon1_x, neon1_Y, neon2_x, neon2_Y, neon3_x, neon3_y, neon4_x, neon4_y)

    if I2 == 0 :
        return 0

    I3 = adjust_corlor(datase_dir, img, I + 0.09, R, r, g, b, neon1_x, neon1_Y, neon2_x, neon2_Y, neon3_x, neon3_y, neon4_x, neon4_y)

    if I3 == 0 :
        return 0

    return 1

# The function of adjusting the radius
def adjust_radius(datase_dir, img, I, R, r, g, b, neon1_x, neon1_Y, neon2_x, neon2_Y, neon3_x, neon3_y, neon4_x, neon4_y):

    radius1 = adjust_intensity(datase_dir, img, I, R - 5, r, g, b, neon1_x, neon1_Y, neon2_x, neon2_Y, neon3_x, neon3_y, neon4_x,
                          neon4_y)

    if radius1 == 0 :
        return 0

    radius2 = adjust_intensity(datase_dir, img, I, R, r, g, b, neon1_x, neon1_Y, neon2_x, neon2_Y, neon3_x, neon3_y, neon4_x,
                          neon4_y)

    if radius2 == 0 :
        return 0

    radius3 = adjust_intensity(datase_dir, img, I, R + 5, r, g, b, neon1_x, neon1_Y, neon2_x, neon2_Y, neon3_x, neon3_y, neon4_x,
                          neon4_y)

    if radius3 == 0 :
        return 0

    return 1

# Adaptive function in the physical environment
def adapt_fuction_physical(datase_dir, img, I, R, r, g, b, neon1_x, neon1_Y, neon2_x, neon2_Y, neon3_x, neon3_y, neon4_x, neon4_y):

    position1 = adjust_radius(datase_dir, img, I, R, r, g, b, neon1_x + 1, neon1_Y, neon2_x + 1, neon2_Y, neon3_x + 1, neon3_y, neon4_x + 1, neon4_y)

    if position1 == 0 :
        return 0

    position2 = adjust_radius(datase_dir, img, I, R, r, g, b, neon1_x - 1, neon1_Y, neon2_x - 1, neon2_Y, neon3_x - 1, neon3_y, neon4_x - 1, neon4_y)

    if position2 == 0 :
        return 0

    position3 = adjust_radius(datase_dir, img, I, R, r, g, b, neon1_x, neon1_Y + 1, neon2_x, neon2_Y + 1, neon3_x, neon3_y + 1, neon4_x, neon4_y + 1)

    if position3 == 0 :
        return 0

    position4 = adjust_radius(datase_dir, img, I, R, r, g, b, neon1_x, neon1_Y - 1, neon2_x, neon2_Y - 1, neon3_x, neon3_y - 1, neon4_x, neon4_y - 1)

    if position4 == 0 :
        return 0

    return 1

#Manually set the optimization range
def optimize_range():

    neon1_x_min,  neon1_x_max = 170, 200
    neon1_y_min,  neon1_y_max = 130, 150
    neon2_x_min,  neon2_x_max = 280, 330
    neon2_y_min,  neon2_y_max = 120, 150
    neon3_x_min,  neon3_x_max = 170, 200
    neon3_y_min,  neon3_y_max = 230, 260
    neon4_x_min,  neon4_x_max = 280, 330
    neon4_y_min,  neon4_y_max = 230, 260

    return neon1_x_min, neon1_x_max, neon1_y_min,  neon1_y_max, neon2_x_min,  neon2_x_max, neon2_y_min,  neon2_y_max, neon3_x_min,  neon3_x_max, neon3_y_min,  neon3_y_max, neon4_x_min,  neon4_x_max, neon4_y_min,  neon4_y_max

#Obtain optimal perturbation
def optimization(tag, text1, text2, text3, text4, text5, text6, text7, text8, text9, text10, text11, text12, dataset_dir1, dataset_dir2, dataset_dir3, dataset_dir4, R, I, confidence1, confidence2, confidence3, neon1_x_optimal_1, neon1_y_optimal_1, neon1_x_optimal_2, neon1_y_optimal_2, neon1_x_optimal_3, neon1_y_optimal_3, neon_x, neon_y, r, g, b):

    img = cv2.imread(dataset_dir1)
    if img.shape is None:
        print("None")
        sys.exit()
    height, width, n = img.shape

    img_neon_effect_physical(R, img, "1", neon_x, neon_y, r, g, b)
    cap = cv2.VideoCapture(dataset_dir1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    i1 = 1
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter('result_p.jpg', fourcc, fps, (width, height))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = video_neon_effect_physical(frame, i1 % 5, I)
            videoWriter.write(frame)
            i1 += 1
            c = cv2.waitKey(1)
            if c == 27:
                break
        else:
            break

    img2 = Image.open('result_p.jpg')
    plt.imshow(img2)
    plt.show()
    label, confidence_919 = classify('result_p.jpg', net)
    label = int(label)
    print("neon1_x = ", neon_x)
    print("neon1_y = ", neon_y)
    print("label = ", label)
    print('confidence_919 = ', confidence_919)
    print('confidence1 = ', confidence1)



    if confidence_919 <= confidence1:

        confidence3 = confidence2
        neon1_x_optimal_3 = neon1_x_optimal_2
        neon1_y_optimal_3 = neon1_y_optimal_2
        confidence2 = confidence1
        neon1_x_optimal_2 = neon1_x_optimal_1
        neon1_y_optimal_2 = neon1_y_optimal_1
        confidence1 = confidence_919
        neon1_x_optimal_1 = neon_x
        neon1_y_optimal_1 = neon_y
        img_save = cv2.imread('result_p.jpg')
        cv2.imwrite(dataset_dir2, img_save)

        if tag == 0 :
            text1 = str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text2 = str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text3 = str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 1 :
            text4 = text1 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text5 = text1 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text6 = text1 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 2 :
            text4 = text2 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text5 = text2 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text6 = text2 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 3 :
            text4 = text3 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text5 = text3 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text6 = text3 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 4 :
            text7 = text4 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text8 = text4 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text9 = text4 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 5 :
            text7 = text5 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text8 = text5 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text9 = text5 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 6 :
            text7 = text6 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text8 = text6 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text9 = text6 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','
        if tag == 7 :
            text10 = text7 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text11 = text7 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text12 = text7 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 8 :
            text10 = text8 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text11 = text8 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text12 = text8 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 9 :
            text10 = text9 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text11 = text9 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text12 = text9 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','


    if  confidence1 < confidence_919 <= confidence2:

        confidence3 = confidence2
        neon1_x_optimal_3 = neon1_x_optimal_2
        neon1_y_optimal_3 = neon1_y_optimal_2
        confidence2 = confidence_919
        neon1_x_optimal_2 = neon_x
        neon1_y_optimal_2 = neon_y
        img_save = cv2.imread('result_p.jpg')
        cv2.imwrite(dataset_dir3, img_save)

        if tag == 0 :
            text2 = str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text3 = str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 1 :
            text4 = text1 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text5 = text1 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text6 = text1 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 2 :
            text4 = text2 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text5 = text2 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text6 = text2 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 3 :
            text4 = text3 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text5 = text3 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text6 = text3 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 4 :
            text7 = text4 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text8 = text4 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text9 = text4 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 5 :
            text7 = text5 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text8 = text5 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text9 = text5 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 6 :
            text7 = text6 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text8 = text6 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text9 = text6 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','
        if tag == 7 :
            text10 = text7 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text11 = text7 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text12 = text7 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 8 :
            text10 = text8 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text11 = text8 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text12 = text8 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 9 :
            text10 = text9 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text11 = text9 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text12 = text9 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','


    if  confidence2 < confidence_919 < confidence3:

        confidence3 = confidence_919
        neon1_x_optimal_3 = neon_x
        neon1_y_optimal_3 = neon_y
        img_save = cv2.imread('result_p.jpg')
        cv2.imwrite(dataset_dir4, img_save)

        if tag == 0 :
            text3 = str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 1 :
            text4 = text1 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text5 = text1 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text6 = text1 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 2 :
            text4 = text2 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text5 = text2 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text6 = text2 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 3 :
            text4 = text3 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text5 = text3 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text6 = text3 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 4 :
            text7 = text4 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text8 = text4 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text9 = text4 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 5 :
            text7 = text5 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text8 = text5 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text9 = text5 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 6 :
            text7 = text6 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text8 = text6 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text9 = text6 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','
        if tag == 7 :
            text10 = text7 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text11 = text7 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text12 = text7 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 8 :
            text10 = text8 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text11 = text8 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text12 = text8 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','

        if tag == 9 :
            text10 = text9 + str(neon1_x_optimal_1) + ',' + str(neon1_y_optimal_1) + ','
            text11 = text9 + str(neon1_x_optimal_2) + ',' + str(neon1_y_optimal_2) + ','
            text12 = text9 + str(neon1_x_optimal_3) + ',' + str(neon1_y_optimal_3) + ','


    print('neon1_x_optimal_1 = ', neon1_x_optimal_1)
    print('neon1_y_optimal_1 = ', neon1_y_optimal_1)
    print('confidence1 = ', confidence1)
    print('neon1_x_optimal_2 = ', neon1_x_optimal_2)
    print('neon1_y_optimal_2 = ', neon1_y_optimal_2)
    print('confidence2 = ', confidence2)
    print('neon1_x_optimal_3 = ', neon1_x_optimal_3)
    print('neon1_y_optimal_3 = ', neon1_y_optimal_3)
    print('confidence3 = ', confidence3)

    print()

    return text1, text2, text3, text4, text5, text6, text7, text8, text9, text10, text11, text12, confidence1, confidence2, confidence3, neon1_x_optimal_1, neon1_y_optimal_1, neon1_x_optimal_2, neon1_y_optimal_2, neon1_x_optimal_3, neon1_y_optimal_3

# label = classify('result/result_digital.jpg', net)
# print('label = ', label)