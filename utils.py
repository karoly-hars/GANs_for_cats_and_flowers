import math
import numpy as np
import cv2


def postprocess_img(img):
    # transform network output into displayable img
    img = img.transpose((1, 2, 0))
    img += 1.0
    img = (img * 128.0).astype(np.uint8)
    return img


def preprocess_img(img):
    # normalize
    img = img / 128.0  # between 0 and 2
    img -= 1.0  # between -1 and 1
    # transpose
    img = img.transpose((2, 0, 1))
    return img


def save_image_grid(img_batch, grid_size, epoch, img_path):
    if grid_size ** 2 != img_batch.shape[0]:
        print("grid_size**2 and batch size not equal: {} {}. Skipping".format(grid_size ** 2, img_batch.shape[0]))
        return None

    # create black canvas
    img_size = img_batch.shape[2]
    canvas = np.zeros((grid_size * (img_size + 2) - 2 + 28, grid_size * (img_size + 2) - 2, 3), dtype=np.uint8)

    # add the epoch number to the bottom
    text_size = cv2.getTextSize("Epoch {}".format(epoch), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)  # get text size
    # calculate text position
    text_left = (canvas.shape[1] - text_size[0][0]) // 2
    text_bottom = canvas.shape[0] - (28 - text_size[0][1]) // 2
    # add text
    cv2.putText(canvas, "Epoch {}".format(epoch), (text_left, text_bottom),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    for img_idx, img in enumerate(img_batch):
        col = math.floor(img_idx / grid_size)
        row = img_idx - col * grid_size
        canvas[col * (img_size + 2): col * (img_size + 2) + img_size,
               row * (img_size + 2): row * (img_size + 2) + img_size, :] = postprocess_img(img)

    cv2.imwrite(img_path, canvas)
