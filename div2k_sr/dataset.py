import os
import shutil
from PIL import Image

import torch.utils.data
import torchvision.transforms as T
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.datasets.utils import download_and_extract_archive


def get_files(root_dir):
    return [os.path.join(root_dir, x) for x in os.listdir(root_dir) if x.endswith(".png")]


class DIV2K(torch.utils.data.Dataset):

    def __init__(self, scale_factor=2, image_size=256, train=True, data_dir="data"):
        self.scale_factor = scale_factor
        self.image_size = image_size
        self.train = train

        self.root_dir = os.path.join(data_dir, "DIV2K")
        if train:
            self.url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
            self.img_dir = os.path.join(self.root_dir, "DIV2K_train_HR")
        else:
            self.url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
            self.img_dir = os.path.join(self.root_dir, "DIV2K_valid_HR")

        self.download()

        self.file_names = get_files(self.img_dir)

        self.lr_transforms = self.get_lr_transforms()
        self.hr_transforms = self.get_hr_transforms()

    def get_lr_transforms(self):
        return Compose(
            [
                Resize(
                    size=(
                        self.image_size // self.scale_factor,
                        self.image_size // self.scale_factor,
                    ),
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                ToTensor(),
            ]
        )

    def get_hr_transforms(self):
        return Compose(
            [
                Resize(
                    (self.image_size, self.image_size),
                    T.InterpolationMode.BICUBIC,
                ),
                ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        lr = Image.open(self.file_names[idx]).convert("RGB")
        hr = lr.copy()
        lr = self.lr_transforms(lr)
        hr = self.hr_transforms(hr)
        return lr, hr

    def __len__(self):
        return len(self.file_names)

    def download(self):
        os.makedirs(self.img_dir, exist_ok=True)
        if not len(get_files(self.img_dir)) > 0:

            if self.train:
                download_and_extract_archive(self.url, self.root_dir, remove_finished=True)
            else:
                download_and_extract_archive(self.url, self.root_dir, remove_finished=True)


def visualize(lr_img, hr_img):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(lr_img.squeeze(0).permute(1, 2, 0))
    ax1.set_title('Low resolution')
    ax1.set_axis_off()
    ax2.imshow(hr_img.squeeze(0).permute(1, 2, 0))
    ax2.set_title('High resolution')
    ax2.set_axis_off()
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    train_dataset = DIV2K()
    train_dataloader = torch.utils.data.DataLoader(train_dataset)
    img_counter = 0
    for lr_img, hr_img in train_dataloader:
        if img_counter > 1:
            break
        visualize(lr_img, hr_img)
        img_counter += 1

    val_dataset = DIV2K(train=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset)
    img_counter = 0
    for lr_img, hr_img in val_dataloader:
        if img_counter > 1:
            break
        visualize(lr_img, hr_img)
        img_counter += 1
