import os
import random
import cv2
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


def load_data(root_dir='data/widerface', split='train'):
    img_path = os.path.join(root_dir, 'WIDER_' + split + '/images')
    anno_path = os.path.join(root_dir, 'wider_face_' + split + '_bbx_gt.txt')

    with open(anno_path, 'rb') as f:
        lines = f.readlines()
    num_lines = len(lines)

    image_data = []
    index = 0
    while index < num_lines:
        filename = lines[index].strip().decode()

        num_obj = int(lines[index + 1])
        filepath = os.path.join(img_path, filename)
        img = cv2.imread(filepath)
        img_height, img_width = img.shape[:2]

        fboxes, iboxes = [], []
        hboxes_f, hboxes_i = [], []
        if num_obj > 0:
            for i in range(num_obj):
                info = lines[index + 2 + i].strip().decode().split(' ')
                x1 = max(int(info[0]), 0)
                y1 = max(int(info[1]), 0)
                w = min(int(info[2]), img_width - x1 - 1)
                h = min(int(info[3]), img_height - y1 - 1)
                box = np.array([x1, y1, x1 + w, y1 + h, 0])
                if w >= 5 and h >= 5:
                    fboxes.append(box)
                    hboxes_f.append(box)
                else:
                    iboxes.append(box)
                    hboxes_i.append(box)
        else:
            num_obj = 1

        fboxes = np.array(fboxes)
        hboxes = np.array(hboxes_f + hboxes_i)
        iboxes = np.array(iboxes)

        annotation = {}
        annotation['filepath'] = filepath
        annotation['fboxes'] = fboxes
        annotation['hboxes'] = hboxes
        annotation['iboxes'] = iboxes
        image_data.append(annotation)
        index += (2 + num_obj)
    return image_data


class WIDERFace(Dataset):
    def __init__(self, path, split, size, transform=None):
        self.dataset = load_data(root_dir=path, split=split)
        self.split = split
        self.transform = transform

        if self.split == 'train':
            random.shuffle(self.dataset)
            self.preprocess = PreProcess(size=size, scale=(16, 32, 64, 128, 256))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        # input is RGB order, and normalized
        img_data = self.dataset[item]
        img = cv2.imread(img_data['filepath'])

        fboxes = img_data['fboxes'].copy()
        hboxes = img_data['hboxes'].copy()
        iboxes = img_data['iboxes'].copy()

        if self.split == 'train':
            img, fboxes, hboxes, iboxes = self.preprocess(
                img, fboxes, hboxes, iboxes)
        if self.transform is not None:
            img = self.transform(img)

        return img, img_data['filepath'], fboxes, hboxes, iboxes


class PreProcess(object):
    """
    Args:
        size: expected output size of each edge
        scale: scale factor
    """
    def __init__(self, size, scale=(16, 32, 64, 128, 256)):
        self.size = size
        self.scale = scale

    def __call__(self, img, fboxes, hboxes, iboxes):
        fboxes = fboxes.copy()
        hboxes = hboxes.copy()
        iboxes = iboxes.copy()

        # random flip
        h, w = img.shape[:2]
        if np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 1)
            if len(fboxes) > 0:
                fboxes[:, [0, 2]] = w - fboxes[:, [2, 0]]
            if len(hboxes) > 0:
                hboxes[:, [0, 2]] = w - hboxes[:, [2, 0]]
            if len(iboxes) > 0:
                iboxes[:, [0, 2]] = w - iboxes[:, [2, 0]]

        # random resize and crop
        img, fboxes, hboxes, iboxes = self.random_resize_crop(
            img, fboxes, hboxes, iboxes, self.size, self.scale, limit=12)

        # random pad
        if np.minimum(img.shape[0], img.shape[1]) < self.size[0]:
            img, fboxes, hboxes, iboxes = self.random_pave(
                img, fboxes, hboxes, iboxes, self.size)

        img = Image.fromarray(img)
        return img, fboxes, hboxes, iboxes

    @staticmethod
    def random_resize_crop(img, fboxes, hboxes, iboxes, size, scales, limit=8):
        h, w = img.shape[:2]
        crop_p, _ = size
        if len(fboxes) > 0:
            sel_id = np.random.randint(0, len(fboxes))
            sel_face = np.sqrt((fboxes[sel_id, 2] - fboxes[sel_id, 0]) *
                               (fboxes[sel_id, 3] - fboxes[sel_id, 1]))
            index = np.random.randint(0, np.argmin(np.abs(scales - sel_face)) + 1)
            sel_tar = np.random.uniform(np.power(2, 4 + index) * 1.5, np.power(2, 4 + index) * 2)
            ratio = sel_tar / sel_face
            n_h, n_w = int(ratio * h), int(ratio * w)
            img = cv2.resize(img, (n_w, n_h))

            crop_x1 = np.random.randint(0, int(fboxes[sel_id, 0]) + 1)
            crop_x1 = np.minimum(crop_x1, np.maximum(0, n_w - crop_p))
            crop_y1 = np.random.randint(0, int(fboxes[sel_id, 1]) + 1)
            crop_y1 = np.minimum(crop_y1, np.maximum(0, n_h - crop_p))
            cropped_img = img[crop_y1:crop_y1 + crop_p, crop_x1:crop_x1 + crop_p]

            # resize label
            fboxes = np.asarray(fboxes, dtype=float)
            fboxes[:, :-1] *= ratio
            if len(hboxes) > 0:
                hboxes = np.asarray(hboxes, dtype=float)
                hboxes[:, :-1] *= ratio
            if len(iboxes) > 0:
                iboxes = np.asarray(iboxes, dtype=float)
                iboxes[:, :-1] *= ratio
        else:
            crop_x1, crop_y1 = 0, 0
            cropped_img = img[0:crop_p, 0:crop_p]

        # crop detections
        if len(fboxes) > 0:
            before_area = (fboxes[:, 2] - fboxes[:, 0]) * (fboxes[:, 3] - fboxes[:, 1])
            fboxes[:, 0:4:2] -= crop_x1
            fboxes[:, 1:4:2] -= crop_y1
            fboxes[:, 0:4:2] = np.clip(fboxes[:, 0:4:2], 0, crop_p)
            fboxes[:, 1:4:2] = np.clip(fboxes[:, 1:4:2], 0, crop_p)
            after_area = (fboxes[:, 2] - fboxes[:, 0]) * (fboxes[:, 3] - fboxes[:, 1])
            keep_inds = ((fboxes[:, 2] - fboxes[:, 0]) >= limit) & \
                        ((fboxes[:, 3] - fboxes[:, 1]) >= limit) & \
                        (after_area >= 0.5 * before_area)
            fboxes[keep_inds, -1] = 1

        if len(hboxes) > 0:
            before_area = (hboxes[:, 2] - hboxes[:, 0]) * (hboxes[:, 3] - hboxes[:, 1])
            hboxes[:, 0:4:2] -= crop_x1
            hboxes[:, 1:4:2] -= crop_y1
            hboxes[:, 0:4:2] = np.clip(hboxes[:, 0:4:2], 0, crop_p)
            hboxes[:, 1:4:2] = np.clip(hboxes[:, 1:4:2], 0, crop_p)
            after_area = (hboxes[:, 2] - hboxes[:, 0]) * (hboxes[:, 3] - hboxes[:, 1])
            keep_inds = ((hboxes[:, 2] - hboxes[:, 0]) >= limit) & \
                        ((hboxes[:, 3] - hboxes[:, 1]) >= limit) & \
                        (after_area >= 0.5 * before_area)
            hboxes[keep_inds, -1] = 1

        if len(iboxes) > 0:
            before_area = (iboxes[:, 2] - iboxes[:, 0]) * (iboxes[:, 3] - iboxes[:, 1])
            iboxes[:, 0:4:2] -= crop_x1
            iboxes[:, 1:4:2] -= crop_y1
            iboxes[:, 0:4:2] = np.clip(iboxes[:, 0:4:2], 0, crop_p)
            iboxes[:, 1:4:2] = np.clip(iboxes[:, 1:4:2], 0, crop_p)
            after_area = (iboxes[:, 2] - iboxes[:, 0]) * (iboxes[:, 3] - iboxes[:, 1])
            keep_inds = ((iboxes[:, 2] - iboxes[:, 0]) >= limit) & \
                        ((iboxes[:, 3] - iboxes[:, 1]) >= limit) & \
                        (after_area < 0.5 * before_area)
            iboxes[keep_inds, -1] = 1
        return cropped_img, fboxes, hboxes, iboxes

    @staticmethod
    def random_pave(img, fboxes, hboxes, iboxes, size, limit=8):
        h, w = img.shape[:2]
        pave_h, pave_w = size
        paved_image = np.ones((pave_h, pave_w, 3), dtype=np.uint8) * np.mean(img, dtype=int)
        pave_x = int(np.random.randint(0, pave_w - w + 1))
        pave_y = int(np.random.randint(0, pave_h - h + 1))
        paved_image[pave_y:pave_y + h, pave_x:pave_x + w] = img

        # pave detections
        if len(fboxes) > 0:
            fboxes[:, 0:4:2] += pave_x
            fboxes[:, 1:4:2] += pave_y
            fboxes[:, -1] = 0
            keep_inds = ((fboxes[:, 2] - fboxes[:, 0]) >= limit) & \
                        ((fboxes[:, 3] - fboxes[:, 1]) >= limit)
            fboxes[keep_inds, -1] = 1

        if len(hboxes) > 0:
            hboxes[:, 0:4:2] += pave_x
            hboxes[:, 1:4:2] += pave_y
            hboxes[:, -1] = 0
            keep_inds = ((hboxes[:, 2] - hboxes[:, 0]) >= limit) & \
                        ((hboxes[:, 3] - hboxes[:, 1]) >= limit)
            hboxes[keep_inds, -1] = 1

        if len(iboxes) > 0:
            iboxes[:, 0:4:2] += pave_x
            iboxes[:, 1:4:2] += pave_y
            iboxes[:, -1] = 0
            keep_inds = ((iboxes[:, 2] - iboxes[:, 0]) >= 8) & \
                        ((iboxes[:, 3] - iboxes[:, 1]) >= 8)
            iboxes[keep_inds, -1] = 1
        return paved_image, fboxes, hboxes, iboxes