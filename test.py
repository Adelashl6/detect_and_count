import torch
import json
import time
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose

from config import Config
from models import DnCNet, get_results
from dataset import make_data_loader
from eval_city.eval_script.eval_demo import validate


def val(cfg, epoch_range):

    # log
    log_file = './log/' + time.strftime('%Y%m%d', time.localtime(time.time())) + '.log.val'
    log = open(log_file, 'w')

    # dataset
    data_loader = make_data_loader(cfg, is_train=False)

    # network
    net = DnCNet(base=cfg.network).cuda()

    for epoch in range(epoch_range[0], epoch_range[1] + 1):
        filename = './ckpt/%s_%d.pth.tea' % ('CSP', epoch)
        teacher_dict = torch.load(filename)
        net.load_state_dict(teacher_dict)
        net.eval()

        print('Perform validation...')
        res = []
        t3 = time.time()
        mae, mae1, mae2, mae3 = 0, 0, 0, 0
        count1, count2, count3 = 0, 0, 0
        for i, data in enumerate(data_loader):
            inputs = data[0].cuda()
            img_path = data[1]
            gt_bboxes = data[2]
            ig_bboxes = data[3]
            with torch.no_grad():
                outputs = net(inputs)

            outputs = [o.cpu().numpy() for o in outputs]
            print(img_path)
            bboxes, counts = get_results(*outputs, cfg.size_test,
                                         ratio=0.41, down=4, nms_thresh=0.5)

            if len(bboxes) > 0:
                bboxes[:, [2, 3]] -= bboxes[:, [0, 1]]
                for bbox in bboxes:
                    temp = dict()
                    temp['image_id'] = i + 1
                    temp['category_id'] = 1
                    temp['bbox'] = bbox[:4].tolist()
                    temp['score'] = float(bbox[4])
                    res.append(temp)
            # vis_detections(img_path[0], bboxes, './result.png')
            error = abs(counts - gt_bboxes.shape[1] - ig_bboxes.shape[1])
            mae += error
            if (gt_bboxes.shape[1] + ig_bboxes.shape[1]) <= 10:
                mae1 += error
                count1 += 1
            elif (gt_bboxes.shape[1] + ig_bboxes.shape[1]) > 10 and (gt_bboxes.shape[1] + ig_bboxes.shape[1]) <= 30:
                mae2 += error
                count2 += 1
            elif (gt_bboxes.shape[1] + ig_bboxes.shape[1]) >= 30:
                mae3 += error
                count3 += 1

        with open('../eval_city/val_pred.json', 'w') as f:
            json.dump(res, f)

        MRs = validate('./eval_city/val_gt.json', '../eval_city/val_pred.json')
        t4 = time.time()
        print('Summarize: [Reasonable: %.2f%%], [Bare: %.2f%%], [Partial: %.2f%%], [Heavy: %.2f%%]'
              % (MRs[0] * 100, MRs[1] * 100, MRs[2] * 100, MRs[3] * 100))
        print('MAE: [Total: %.2f], [Num <= 10: %.2f], [10 < Num <= 30: %.2f], [Num >= 30: %.2f]'
              % (mae / 500, mae1 / count1, mae2 / count2, mae3 / count3))
        if log is not None:
            log.write("%d %.7f %.7f %.7f %.7f\n" % (epoch, MRs[0], MRs[1], MRs[2], MRs[3]))
        print('Validation time used: %.3f' % (t4 - t3))

    log.close()


if __name__ == '__main__':
    cfg = Config()
    val_epoch_range = [97, 97]
    val(cfg, val_epoch_range)