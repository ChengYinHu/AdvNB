import cv2
import matplotlib.pyplot as plt
import torch
from neonlight_simulation import img_neon_effetc_digital, video_neon_effect_digital, classify
import torchvision.models as models
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = models.resnet50(pretrained=True).eval().to(device)

count = 1

# The number of neon beams
N = random.randint(5, 10)
# The radius of the neon beam
R = random.randint(3, 6)
# The intensity of the neon beam
I = random.randint(1, 2)

if __name__ == "__main__":

    for step in range(0, 2000):
        print('step = ', step)

        img_path = 'dataset/2.jpg'
        img = cv2.imread(img_path)
        height, width, n = img.shape

        N = random.randint(20, 30)  # The number of neon beams
        print('N = ', N)
        for beam_number in range(0, N):
            R = random.randint(10, 20)  # The radius of the neon beam
            I = random.randint(1, 3) / 10  # The intensity of the neon beam
            x = random.randint(R, width - R)  # The x axis of the neon beam
            y = random.randint(R, height - R)  # The y axis of the neon beam
            r = random.randint(0, 255)  # The red channel of the neon beam color
            g = random.randint(0, 255)  # The green channel of the neon beam color
            b = random.randint(0, 255)  # The blue channel of the neon beam color
            img_neon_effetc_digital(img, x, y, r, g, b, '1', R)

        cap = cv2.VideoCapture(img_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        i = 1
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        videoWriter = cv2.VideoWriter('result/result_digital.jpg', fourcc, fps, (width, height))
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frame = video_neon_effect_digital(frame, i % 5, I)
                # cv2.imshow('video', frame)
                videoWriter.write(frame)
                i += 1
                c = cv2.waitKey(1)
                if c == 27:
                    break
            else:
                break

        img_clean = plt.imread(img_path)
        # plt.imshow(img_clean)
        # plt.show()

        label_clean, Confidence_919 = classify(img_path, net)
        print('label_clean : ', label_clean)

        img_adv = plt.imread('result/result_digital.jpg')
        # plt.imshow(img_adv)
        # plt.show()

        label_adv, Confidence_919 = classify('result/result_digital.jpg', net)
        print('label_adv : ', label_adv)

        label_clean = int(label_clean)
        label_adv = int(label_adv)

        if label_adv != label_clean:
            break
