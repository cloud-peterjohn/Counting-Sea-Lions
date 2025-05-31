import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import skimage.feature


def get_data(basepath, filename, patch_size = 40, visualize=False):
    # read the Train and Train Dotted images
    image_1 = cv2.imread(os.path.join(basepath, "TrainDotted", filename))
    image_2 = cv2.imread(os.path.join(basepath, "Train", filename))
    img1 = cv2.GaussianBlur(image_1, (5, 5), 0)

    # absolute difference between Train and Train Dotted
    image_3 = cv2.absdiff(image_1, image_2)
    mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    mask_1[mask_1 < 50] = 0
    mask_1[mask_1 > 0] = 255
    image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)

    # convert to grayscale to be accepted by skimage.feature.blob_log
    image_6 = np.max(image_4, axis=2)

    # detect blobs
    blobs = skimage.feature.blob_log(
        image_6, min_sigma=3, max_sigma=7, num_sigma=1, threshold=0.05
    )

    h, w, d = image_2.shape

    res = np.zeros((int(w // patch_size) + 1, int(h // patch_size) + 1, 5), dtype="int16")

    for blob in blobs:
        # get the coordinates for each blob
        y, x, s = blob
        # get the color of the pixel from Train Dotted in the center of the blob
        b, g, R = img1[int(y)][int(x)][:]
        x1 = int(x // patch_size)
        y1 = int(y // patch_size)
        # decision tree to pick the class of the blob by looking at the color in Train Dotted
        if R > 225 and b < 25 and g < 25:  # RED
            res[x1, y1, 0] += 1
        elif R > 225 and b > 225 and g < 25:  # MAGENTA
            res[x1, y1, 1] += 1
        elif R < 75 and b < 50 and 150 < g < 200:  # GREEN
            res[x1, y1, 4] += 1
        elif R < 75 and 150 < b < 200 and g < 75:  # BLUE
            res[x1, y1, 3] += 1
        elif 60 < R < 120 and b < 50 and g < 75:  # BROWN
            res[x1, y1, 2] += 1

    ma = cv2.cvtColor(
        (1 * (np.sum(image_1, axis=2) > 20)).astype("uint8"), cv2.COLOR_GRAY2BGR
    )
    img = image_2 * ma
    h1, w1, d = img.shape

    trainX = []
    trainY = []

    for i in range(int(w1 // patch_size)):
        for j in range(int(h1 // patch_size)):
            trainY.append(res[i, j, :])
            trainX.append(
                img[j * patch_size : j * patch_size + patch_size, i * patch_size : i * patch_size + patch_size, :]
            )

    trainX, trainY = np.array(trainX), np.array(trainY)

    if not visualize:
        return trainX, trainY

    # visualize the result
    vis_img = img.copy()
    mask = np.zeros_like(vis_img)

    colors = [
        (255, 0, 0),  # RED
        (255, 0, 255),  # MAGENTA
        (42, 42, 165),  # BROWN
        (255, 165, 0),  # BLUE
        (0, 255, 0),  # GREEN
    ]

    idx = 0
    for i in range(int(w1 // patch_size)):
        for j in range(int(h1 // patch_size)):
            y0, y1_ = j * patch_size, min((j + 1) * patch_size, h1)
            x0, x1_ = i * patch_size, min((i + 1) * patch_size, w1)
            for c in range(5):
                if trainY[idx][c] > 0:
                    mask[y0:y1_, x0:x1_] = colors[c]
            idx += 1

    alpha = 0.4
    vis = cv2.addWeighted(vis_img, 1 - alpha, mask, alpha, 0)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title("Image with Mask Overlay")
    plt.axis("off")
    plt.show()

    return trainX, trainY


x, y = get_data("TrainSmall2", "41.jpg", visualize=True)
