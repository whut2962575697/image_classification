# -*- encoding: utf-8 -*-
'''
@File    :   inference.py
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/2/26 8:37   xin      1.0         None
'''
import torch
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm
from common import tta
from common.tta import ClassificationTTAWrapper


def inference_val(dataloader, model, device, use_tta=True):
    model = model.to(device)
    model.eval()

    if use_tta:
        tta_transforms = tta.Compose(
            [
                tta.FiveCrops(32, 32),
                tta.HorizontalFlip(),


            ]
        )
        model = ClassificationTTAWrapper(model, tta_transforms)

    id = 0
    preds_list = list()
    all_outputs = list()
    all_targets = list()
    for batch_data in tqdm(dataloader, total=len(dataloader), leave=False):

        images, labels = batch_data

        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            batch_pred = list(outputs.argmax(dim=1).cpu().numpy())
            for y_pred in batch_pred:
                preds_list.append((id, y_pred))
                id += 1
        labels = labels.data.cpu()
        outputs = outputs.data.cpu()
        all_outputs.append(outputs)
        all_targets.append(labels)
    all_outputs = torch.cat(all_outputs, 0)
    all_targets = torch.cat(all_targets, 0)
    acc = accuracy_score(np.argmax(all_outputs, 1), all_targets)
    print("val acc: {0}".format((acc)))

    print('生成提交结果文件')
    with open('submission1.csv', 'w') as f:
        f.write('ID,Prediction\n')
        for id, pred in preds_list:
            f.write('{},{}\n'.format(id, pred))



if __name__ == "__main__":

    from config import cfg
    from model import build_model
    from common.sync_bn import convert_model
    from dataset.data import Fashion_MNIST_DataSet



    import torchvision
    import torchvision.transforms as T
    import albumentations as A
    from torch.utils.data.dataloader import DataLoader

    device = torch.device("cpu")
    num_gpus = 0
    if cfg.MODEL.DEVICE == 'cuda' and torch.cuda.is_available():
        num_gpus = len(cfg.MODEL.DEVICE_IDS) - 1
        device_ids = cfg.MODEL.DEVICE_IDS.strip("d")
        print(device_ids)
        device = torch.device("cuda:{0}".format(device_ids))

    model = build_model(cfg)
    para_dict = torch.load(r'/usr/demo/common_data/baseline_epoch363.pth')

    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    if cfg.SOLVER.SYNCBN:
        model = convert_model(model)
    model.load_state_dict(para_dict)

    main_transform = A.Compose \
            ([
            A.Resize(cfg.INPUT.RESIZE_TEST[0], cfg.INPUT.RESIZE_TEST[1]),
            # A.CenterCrop(cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1])
        ])
    image_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])

    dataset = Fashion_MNIST_DataSet(cfg, mode='val', main_transform=main_transform,
                                        img_transform=image_transform)
    dataloader = DataLoader(
        dataset, batch_size=cfg.SOLVER.PER_BATCH, shuffle=False,
        num_workers=2
    )

    inference_val(dataloader, model, device)








