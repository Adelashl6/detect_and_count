import torch
import numpy as np

from torchvision.transforms import ToTensor, Normalize, Compose, ColorJitter
from dataset import CityPersons, WIDERFace, CrowdHuman


def train_collater(batch):
    batch_size = len(batch)
    transposed_batch = list(zip(*batch))
    results = {}

    imgs = transposed_batch[0]
    paths = transposed_batch[1]
    fboxes = transposed_batch[2]
    hboxes = transposed_batch[3]
    iboxes = transposed_batch[4]

    widths = [int(s.shape[2]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()
    padded = np.zeros((batch_size, 3, max_height, max_width))
    for i in range(batch_size):
        img = imgs[i]
        padded[i, :, :img.shape[1], :img.shape[2]] = img
    results['imgs'] = torch.tensor(padded, dtype=torch.float32)

    max_num = max(boxes.shape[0] for boxes in fboxes)
    padded = np.ones((len(fboxes), max_num, 5)) * -1
    if max_num > 0:
        for idx, boxes in enumerate(fboxes):
            if boxes.shape[0] > 0:
                padded[idx, :boxes.shape[0], :] = boxes
    else:
        padded = np.ones((len(fboxes), 1, 5)) * -1
    results['fboxes'] = torch.tensor(padded, dtype=torch.float32)

    max_num = max(boxes.shape[0] for boxes in hboxes)
    padded = np.ones((len(hboxes), max_num, 5)) * -1
    if max_num > 0:
        for idx, boxes in enumerate(hboxes):
            if boxes.shape[0] > 0:
                padded[idx, :boxes.shape[0], :] = boxes
    else:
        padded = np.ones((len(hboxes), 1, 5)) * -1
    results['hboxes'] = torch.tensor(padded, dtype=torch.float32)

    max_num = max(boxes.shape[0] for boxes in iboxes)
    padded = np.ones((len(iboxes), max_num, 5)) * -1
    if max_num > 0:
        for idx, boxes in enumerate(iboxes):
            if boxes.shape[0] > 0:
                padded[idx, :boxes.shape[0], :] = boxes
    else:
        padded = np.ones((len(iboxes), 1, 5)) * -1
    results['iboxes'] = torch.tensor(padded, dtype=torch.float32)

    results['paths'] = paths
    return results


def test_collater(batch):
    batch_size = 1
    transposed_batch = list(zip(*batch))
    results = {}

    imgs = transposed_batch[0]
    paths = transposed_batch[1]
    fboxes = transposed_batch[2]
    hboxes = transposed_batch[3]
    iboxes = transposed_batch[4]

    results['imgs'] = torch.tensor(imgs, dtype=torch.float32)
    results['paths'] = paths
    results['fboxes'] = torch.tensor(fboxes, dtype=torch.float32)
    results['hboxes'] = torch.tensor(hboxes, dtype=torch.float32)
    results['iboxes'] = torch.tensor(iboxes, dtype=torch.float32)
    return results


def build_dataset(cfg, transforms=None, is_train=True):
    if is_train:
        split = 'train'
        size = cfg.size_train
    else:
        split = 'val'
        size = cfg.size_train

    if cfg.datasets == 'citypersons':
        dataset = CityPersons(path=cfg.path,
                              split=split,
                              size=size,
                              transform=transforms)
    elif cfg.datasets == 'widerface':
        dataset = WIDERFace(path=cfg.path,
                            split=split,
                            size=size,
                            transform=transforms)
    elif cfg.datasets == 'crowdhuman':
        dataset = CrowdHuman(path=cfg.path,
                             split=split,
                             size=size,
                             transform=transforms)
    else:
        raise NotImplementedError
    return dataset


def make_data_loader(cfg, is_train=True):
    if is_train:
        batch_size = cfg.imgs_per_batch * len(cfg.gpu_ids)
        transforms = Compose([
            ColorJitter(brightness=0.5),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        collator = train_collater
    else:
        batch_size = 1
        transforms = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        collator = test_collater

    dataset = build_dataset(cfg, transforms, is_train)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=batch_size)
    return data_loader

