import os
import mimetypes
import torchvision
from facenet_pytorch import MTCNN

import numpy as np
import cv2
import torch
from facenet_pytorch.models.utils.detect_face import save_img
from torch.nn.functional import interpolate
from PIL import Image


def get_size(img):
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return img.shape[1::-1]
    else:
        return img.size


def extract_face(img, box, image_size=160, margin=[0, 0], save_path=None):
    """Extract face + margin from PIL Image given bounding box.

    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted face image. (default: {None})

    Returns:
        torch.tensor -- tensor representing the extracted face.
    """
    box[1] = box[1] - margin[0]
    box[3] = box[3] + margin[0]
    box[0] = box[0] - margin[1]
    box[2] = box[2] + margin[1]
    face = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
        save_img(face, save_path)

    return face, box


def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data


def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = cv2.resize(
            img,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA
        ).copy()
    elif isinstance(img, torch.Tensor):
        img = img[box[1]:box[3], box[0]:box[2]]
        out = imresample(
            img.permute(2, 0, 1).unsqueeze(0).float(),
            (image_size, image_size)
        ).byte().squeeze(0).permute(1, 2, 0)
    else:
        out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    return out

def is_video_format(filename):
    mimestart = mimetypes.guess_type(filename)[0]
    if mimestart is None:
        return False
    mimestart = mimestart.split('/')[0]
    if mimestart not in ['video']:
        return False
    return True


def crop_images(select_folder, new_folder):
    for dirname, _, filenames in os.walk(select_folder):
        for filename in filenames:
            if not is_video_format(filename):
                continue
            video_path = os.path.join(dirname, filename)
            frames = torchvision.io.read_video(video_path)[0]
            for f_id, frame in enumerate(frames):
                img_tensor = frame.numpy()
                img = Image.fromarray(img_tensor)
                mtcnn = MTCNN(image_size=224)
                box = mtcnn.detect(img)
                save_fname = f"{filename.split('.')[0]}_frame_{f_id}.jpg"
                save_path = os.path.join(new_folder, save_fname)
                print(save_path)
                extract_face(img, box[0][0], save_path=save_path)

if __name__ == "__main__":
    mimetypes.init()
    crop_images("data/face_detection", "data/cropped_faces")
