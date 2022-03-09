import torch
import os.path as osp

class Config(object):
    def __init__(self):
        self.gpu_ids = [0, 1]
        self.imgs_per_batch = 8
        self.alpha = 0.999
        self.sigma = 0.5

        # network architechture
        self.network = 'resnet50'  # or 'dla34'
        self.down = 4  # downsampling rate of the feature map for detection
        self.radius = 2  # surrounding areas of positives for the scale map

        # config
        self.teacher = True
        self.device = torch.device("cuda")

        # citypersons
        # self.datasets = 'citypersons'
        # self.path = osp.join('./data', self.datasets)
        # self.num_epochs = 150
        # self.init_lr = 1e-4
        # self.size_train = (640, 1280)
        # self.size_test = (1024, 2048)
        '''
        # crowdhuman
        self.datasets = 'crowdhuman'
        self.path = osp.join('./data', self.datasets)
        self.num_epochs = 150
        self.init_lr = 1e-4
        self.size_train = (768, 1152)
        self.size_test = (1024, 2048)
        '''
        # widerface
        self.datasets = 'widerface'
        self.path = osp.join('./data', self.datasets)
        self.num_epochs = 400
        self.init_lr = 1e-4
        self.size_train = (704, 704)
        self.size_test = (1024, 2048)

    def print_conf(self):
        print ('\n'.join(['%s:%s' % item for item in self.__dict__.items()]))