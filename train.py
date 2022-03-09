import os
import time
import torch
import torch.optim as optim

from config import Config
from models import (cls_loss, reg_loss, off_loss, den_loss, dnc_loss,
                        DnCNet, get_targets)
from dataset import make_data_loader


def train(cfg):

    if not os.path.exists('./ckpt'):
        os.mkdir('./ckpt')
    if not os.path.exists('./log'):
        os.mkdir('./log')

    # log
    log_file = './log/' + time.strftime('%Y%m%d', time.localtime(time.time())) + '.log'
    log = open(log_file, 'w')

    # dataset
    data_loader = make_data_loader(cfg, is_train=True)

    # network
    net = DnCNet(base=cfg.network).cuda()

    # continue training
    net.load_state_dict(torch.load('./ckpt/CSP_291.pth.tea'))

    # optimizer
    params = []
    for n, p in net.named_parameters():
        if p.requires_grad:
            params.append({'params': p})

    # use teacher network
    if cfg.teacher:
        teacher_dict = net.state_dict()

    # multi-gpu
    if len(cfg.gpu_ids) > 1:
        net = torch.nn.DataParallel(net, device_ids=cfg.gpu_ids)
    optimizer = optim.Adam(params, lr=cfg.init_lr)
    num_batch_per_epoch = len(data_loader)
    cfg.print_conf()

    # loss
    loss_cls = cls_loss().cuda()
    loss_reg = reg_loss().cuda()
    loss_off = off_loss().cuda()
    loss_den = den_loss().cuda()
    # loss_dnc = dnc_loss(cfg.size_train, cfg.down).cuda()

    for epoch in range(292, cfg.num_epochs):
        print('Epoch %d begin' % (epoch + 1))
        t1 = time.time()

        epoch_loss = 0.0
        net.train()

        for i, data in enumerate(data_loader):
            t3 = time.time()
            # get the inputs
            imgs = data["imgs"].to(cfg.device)
            paths = data["paths"]
            fboxes = data['fboxes'].to(cfg.device)
            hboxes = data['hboxes'].to(cfg.device)
            iboxes = data['iboxes'].to(cfg.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # heat map
            cls, reg, off, den = net(imgs)
            # generate target
            featmap_size = cls.size()[-2:]
            targets = get_targets(fboxes, hboxes, iboxes, featmap_size, cfg)
            (cls_targets, reg_targets, off_targets, den_targets,
             region_mask_list, post_prob_list) = targets

            reg_weights = [(region_mask > 0).sum(0, keepdim=True)
                           for region_mask in region_mask_list]
            reg_weights = torch.stack(reg_weights, dim=0).float()

            off_weights = [(region_mask == 1).sum(0, keepdim=True)
                           for region_mask in region_mask_list]
            off_weights = torch.stack(off_weights, dim=0).float()

            ctr_mask_list = [(region_mask > 0).float()
                             for region_mask in region_mask_list]

            # loss
            cls_loss_value = loss_cls(cls, cls_targets, factor=0.01)
            reg_loss_value = loss_reg(reg, reg_targets, reg_weights, factor=1.0)
            off_loss_value = loss_off(off, off_targets, off_weights, factor=0.1)
            den_loss_value = loss_den(den, den_targets, factor=0.01)
            '''
            dnc_loss_value = loss_dnc(cls, reg, off, den, fboxes, hboxes,
                                      ctr_mask_list, post_prob_list, factor=0.1)
            '''
            loss = cls_loss_value + reg_loss_value + off_loss_value + den_loss_value

            # back-prop
            loss.backward()
            # update param
            optimizer.step()

            if cfg.teacher:
                # for k, v in net.module.state_dict().items():
                for k, v in net.module.state_dict().items():
                    if k.find('num_batches_tracked') == -1:  # skip bn layer
                        teacher_dict[k] = cfg.alpha * teacher_dict[k] + (1 - cfg.alpha) * v
                    else:
                        teacher_dict[k] = 1 * v

            # print statistics
            t4 = time.time()
            print('\r[Epoch %d/%d, Batch %d/%d] <Total: %.6f> '
                  'cls: %.6f, reg: %.6f, off: %.6f, den: %.6f,  Time: %.3f sec' %
                  (epoch + 1, cfg.num_epochs, i + 1, num_batch_per_epoch,
                   loss.item(), cls_loss_value.item(), reg_loss_value.item(),
                   off_loss_value.item(), den_loss_value.item(), t4 - t3))
            epoch_loss += loss.item()
        print('')

        t2 = time.time()
        epoch_loss /= num_batch_per_epoch
        print('Epoch %d end, AvgLoss is %.6f, Time used %.1f sec.' % (epoch + 1, epoch_loss, int(t2 - t1)))
        log.write('%d %.7f\n' % (epoch + 1, epoch_loss))

        print('Save checkpoint...')
        filename = './ckpt/%s_%d.pth' % ('CSP', epoch + 1)
        torch.save(net.module.state_dict(), filename)
        if cfg.teacher:
            torch.save(teacher_dict, filename + '.tea')
        print('%s saved.' % filename)
    log.close()


if __name__ == '__main__':
    cfg = Config()
    train(cfg)