import os
import random
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from scipy import io as scio


def load_data(root_dir='data/citypersons', split='train'):
    all_img_path = os.path.join(root_dir, 'images')
    all_anno_path = os.path.join(root_dir, 'annotations')
    rows, cols = 1024, 2048

    anno_path = os.path.join(all_anno_path, 'anno_' + split + '.mat')
    res_path = os.path.join('data/cache/cityperson', split)
    annos = scio.loadmat(anno_path)
    index = 'anno_' + split + '_aligned'

    image_data = []
    for l in range(len(annos[index][0])):
        anno = annos[index][0][l]
        cityname = anno[0][0][0][0]
        imgname = anno[0][0][1][0]
        gts = anno[0][0][2]
        img_path = os.path.join(all_img_path, split + '/' + cityname + '/' + imgname)

        fboxes, iboxes = [], []
        hboxes_f, hboxes_i = [], []
        for i in range(len(gts)):
            label, x1, y1, w, h = gts[i, :5]
            xv1, yv1, wv, hv = gts[i, 6:]

            x1, y1 = max(x1, 0), max(y1, 0)
            w, h = min(w, cols - x1 - 1), min(h, rows - y1 - 1)
            xv1, yv1 = max(xv1, 0), max(yv1, 0)
            wv, hv = min(wv, cols - xv1 - 1), min(hv, rows - yv1 - 1)
            visibility = (wv * hv) / (w * h)

            box = np.array([x1, y1, x1 + w, y1 + h, 0])
            hbox = np.array([x1 + 0.25 * w, y1, x1 + 0.75 * w, y1 + 0.2 * h, 0])
            if label == 1 and h >= 50 and visibility >= 0:
                fboxes.append(box)
                hboxes_f.append(hbox)
            else:
                iboxes.append(box)
                hboxes_i.append(hbox)

        fboxes = np.array(fboxes)
        hboxes = np.array(hboxes_f + hboxes_i)
        iboxes = np.array(iboxes)

        annotation = {}
        annotation['filepath'] = img_path
        annotation['fboxes'] = fboxes
        annotation['hboxes'] = hboxes
        annotation['iboxes'] = iboxes
        image_data.append(annotation)
    return image_data


class CityPersons(Dataset):
    def __init__(self, path, split, size, transform=None):
        self.dataset = load_data(root_dir=path, split=split)
        self.split = split
        self.transform = transform

        if self.split == 'train':
            random.shuffle(self.dataset)
            self.preprocess = PreProcess(size=size, scale=(0.4, 1.5))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        # input is RGB order, and normalized
        img_data = self.dataset[item]
        img = Image.open(img_data['filepath'])

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
        interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, scale=(0.4, 1.5), interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.scale = scale

    def __call__(self, img, fboxes, hboxes, iboxes):
        # resize image
        w, h = img.size
        ratio = np.random.uniform(self.scale[0], self.scale[1])
        n_w, n_h = int(ratio * w), int(ratio * h)
        img = img.resize((n_w, n_h), self.interpolation)

        fboxes = fboxes.copy()
        hboxes = hboxes.copy()
        iboxes = iboxes.copy()

        # resize label
        if len(fboxes) > 0:
            fboxes = np.asarray(fboxes, dtype=float)
            fboxes[:, :-1] *= ratio
        if len(hboxes) > 0:
            hboxes = np.asarray(hboxes, dtype=float)
            hboxes[:, :-1] *= ratio
        if len(iboxes) > 0:
            iboxes = np.asarray(iboxes, dtype=float)
            iboxes[:, :-1] *= ratio

        # random flip
        if np.random.randint(0, 2) == 0:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if len(fboxes) > 0:
                fboxes[:, [0, 2]] = n_w - fboxes[:, [2, 0]]
            if len(hboxes) > 0:
                hboxes[:, [0, 2]] = n_w - hboxes[:, [2, 0]]
            if len(iboxes) > 0:
                iboxes[:, [0, 2]] = n_w - iboxes[:, [2, 0]]

        if n_h >= self.size[0]:
            # random crop
            img, fboxes, hboxes, iboxes = self.random_crop(
                img, fboxes, hboxes, iboxes, self.size, limit=16)
        else:
            # random pad
            img, fboxes, hboxes, iboxes = self.random_pave(
                img, fboxes, hboxes, iboxes, self.size, limit=16)
        return img, fboxes, hboxes, iboxes

    @staticmethod
    def random_crop(img, fboxes, hboxes, iboxes, size, limit=8):
        w, h = img.size
        crop_h, crop_w = size
        if len(fboxes) > 0:
            sel_id = np.random.randint(0, len(fboxes))
            sel_center_x = int((fboxes[sel_id, 0] + fboxes[sel_id, 2]) / 2.0)
            sel_center_y = int((fboxes[sel_id, 1] + fboxes[sel_id, 3]) / 2.0)
        else:
            sel_center_x = int(np.random.randint(0, w - crop_w + 1) + crop_w * 0.5)
            sel_center_y = int(np.random.randint(0, h - crop_h + 1) + crop_h * 0.5)

        crop_x1 = max(sel_center_x - int(crop_w * 0.5), int(0))
        crop_y1 = max(sel_center_y - int(crop_h * 0.5), int(0))
        diff_x = max(crop_x1 + crop_w - w, int(0))
        crop_x1 -= diff_x
        diff_y = max(crop_y1 + crop_h - h, int(0))
        crop_y1 -= diff_y
        cropped_img = img.crop((crop_x1, crop_y1, crop_x1 + crop_w, crop_y1 + crop_h))

        # crop detections
        if len(fboxes) > 0:
            before_area = (fboxes[:, 2] - fboxes[:, 0]) * (fboxes[:, 3] - fboxes[:, 1])
            fboxes[:, 0:4:2] -= crop_x1
            fboxes[:, 1:4:2] -= crop_y1
            fboxes[:, 0:4:2] = np.clip(fboxes[:, 0:4:2], 0, crop_w)
            fboxes[:, 1:4:2] = np.clip(fboxes[:, 1:4:2], 0, crop_h)
            after_area = (fboxes[:, 2] - fboxes[:, 0]) * (fboxes[:, 3] - fboxes[:, 1])
            keep_inds = ((fboxes[:, 2] - fboxes[:, 0]) >= limit) & (after_area >= 0.5 * before_area)
            fboxes[keep_inds, -1] = 1

        if len(hboxes) > 0:
            before_area = (hboxes[:, 2] - hboxes[:, 0]) * (hboxes[:, 3] - hboxes[:, 1])
            hboxes[:, 0:4:2] -= crop_x1
            hboxes[:, 1:4:2] -= crop_y1
            hboxes[:, 0:4:2] = np.clip(hboxes[:, 0:4:2], 0, crop_w)
            hboxes[:, 1:4:2] = np.clip(hboxes[:, 1:4:2], 0, crop_h)
            after_area = (hboxes[:, 2] - hboxes[:, 0]) * (hboxes[:, 3] - hboxes[:, 1])
            keep_inds = (after_area >= 0.5 * before_area)
            hboxes[keep_inds, -1] = 1

        if len(iboxes) > 0:
            # before_area = (iboxes[:, 2] - iboxes[:, 0]) * (iboxes[:, 3] - iboxes[:, 1])
            iboxes[:, 0:4:2] -= crop_x1
            iboxes[:, 1:4:2] -= crop_y1
            iboxes[:, 0:4:2] = np.clip(iboxes[:, 0:4:2], 0, crop_w)
            iboxes[:, 1:4:2] = np.clip(iboxes[:, 1:4:2], 0, crop_h)
            keep_inds = ((iboxes[:, 2] - iboxes[:, 0]) >= 8)
            iboxes[keep_inds, -1] = 1
        return cropped_img, fboxes, hboxes, iboxes

    @staticmethod
    def random_pave(img, fboxes, hboxes, iboxes, size, limit=8):
        w, h = img.size
        pave_h, pave_w = size
        paved_image = np.ones((pave_h, pave_w, 3), dtype=np.uint8) * np.mean(img, dtype=int)
        pave_x = int(np.random.randint(0, pave_w - w + 1))
        pave_y = int(np.random.randint(0, pave_h - h + 1))
        paved_image[pave_y:pave_y + h, pave_x:pave_x + w] = img
        paved_image = Image.fromarray(paved_image)

        # pave detections
        if len(fboxes) > 0:
            before_area = (fboxes[:, 2] - fboxes[:, 0]) * (fboxes[:, 3] - fboxes[:, 1])
            fboxes[:, 0:4:2] += pave_x
            fboxes[:, 1:4:2] += pave_y
            keep_inds = ((fboxes[:, 2] - fboxes[:, 0]) >= limit)
            fboxes[keep_inds, -1] = 1

        if len(hboxes) > 0:
            before_area = (hboxes[:, 2] - hboxes[:, 0]) * (hboxes[:, 3] - hboxes[:, 1])
            hboxes[:, 0:4:2] += pave_x
            hboxes[:, 1:4:2] += pave_y
            after_area = (hboxes[:, 2] - hboxes[:, 0]) * (hboxes[:, 3] - hboxes[:, 1])
            keep_inds = (after_area >= 0.5 * before_area)
            hboxes[keep_inds, -1] = 1

        if len(iboxes) > 0:
            # before_area = (iboxes[:, 2] - iboxes[:, 0]) * (iboxes[:, 3] - iboxes[:, 1])
            iboxes[:, 0:4:2] += pave_x
            iboxes[:, 1:4:2] += pave_y
            keep_inds = ((iboxes[:, 2] - iboxes[:, 0]) >= 8)
            iboxes[keep_inds, -1] = 1
        return paved_image, fboxes, hboxes, iboxes