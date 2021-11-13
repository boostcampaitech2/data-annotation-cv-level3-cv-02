import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from dataset import CamperTextDataset
from model import EAST
from torch.utils.data import ConcatDataset, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import random
from importlib import import_module

from utils import Wandb,Calc_eval
from utils import make_run_id, make_dir, save_args_to_json
from transform import BasicTransform


EVAL_DATA_LIST = {
    'loss': 0.0,
    'Cls loss': 0.0,
    'Angle loss': 0.0,
    'IoU loss': 0.0
}

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str, nargs="+",
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--train_transform', type=str, default="DefaultTransform")
    parser.add_argument('--valid_transform', type=str, default="DefaultTransform")

    parser.add_argument('--wandb_env_path', type=str, default="./.env")
    parser.add_argument('--wandb_entity', type=str, default="boostcamp-2th-cv-02team")
    parser.add_argument('--wandb_project', type=str, default="data-annotation-cv-level3-02")
    parser.add_argument('--wandb_unique_tag', type=str, default="")
   
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_validation(model, valid_loader):
    model.eval()

    calc_eval_per_epoch = Calc_eval(**EVAL_DATA_LIST)
    with torch.no_grad():
        for img, gt_score_map, gt_geo_map, roi_mask in valid_loader:
            loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
            loss_val = loss.item()
            val_dict = {
                'loss': loss_val, 'Cls loss': extra_info['cls_loss'], 
                'Angle loss': extra_info['angle_loss'], 'IoU loss': extra_info['iou_loss']
            }
            calc_eval_per_epoch.input_data(**val_dict)
    valid_metrics = calc_eval_per_epoch.get_result_data()

    return valid_metrics


def train_val_split(dataset, val_split=0.2):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, random_state=42)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def do_training(wandb, cur_path, data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, seed, train_transform, valid_transform, wandb_env_path, wandb_entity,wandb_project,wandb_unique_tag) :
     # init seed
    set_seed(args.seed)
    
    # data_dir로 복수개의 경로가 입력 되었을 때, 혹은 단일 경로만 입력 되었을 때를 처리하기 위함
    if type(data_dir) is not list:
        data_dir = [data_dir]

    train_transform = getattr(import_module("transform"),args.train_transform)()
    valid_transform = getattr(import_module("transform"),args.valid_transform)()
    
    # load train data
    dataset = [SceneTextDataset(x, split='train', image_size=image_size, crop_size=input_size, transform=train_transform) for x in data_dir]
    dataset = EASTDataset(ConcatDataset(dataset))

    # load validation data
    valset = EASTDataset(SceneTextDataset("../input/data/ICDAR17_Korean", split="val", image_size=image_size, crop_size=input_size, transform=valid_transform))
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    best_metric_score = int(1e9)

    for epoch in range(max_epoch):
        if epoch % 10 ==0:
            print(f"---CURRENT {epoch}EPOCH---")
        calc_eval_per_epoch = Calc_eval(**EVAL_DATA_LIST)
        model.train()
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'loss':loss_val, 'Cls loss': extra_info['cls_loss'], 
                    'Angle loss': extra_info['angle_loss'], 'IoU loss': extra_info['iou_loss']
                }
                calc_eval_per_epoch.input_data(**val_dict)
                pbar.set_postfix(val_dict)
        train_eval_metric = calc_eval_per_epoch.get_result_data()
        wandb.log("Train", **train_eval_metric)

        valid_eval_metric = do_validation(model, valid_loader)
        wandb.log("Valid", **valid_eval_metric)
        scheduler.step()

        print('Mean loss: {:.4f} | Val loss: {:.4f} | Elapsed time: {} '.format(
            epoch_loss / num_batches, valid_eval_metric['loss'], timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(cur_path):
                os.makedirs(cur_path)

            ckpt_fpath = osp.join(cur_path, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            
        if valid_eval_metric['loss'] < best_metric_score:
            if not osp.exists(cur_path):
                os.makedirs(cur_path)

            ckpt_fpath = osp.join(cur_path, 'best.pth')
            torch.save(model.state_dict(), ckpt_fpath)
            best_metric_score = valid_eval_metric['loss']
            wandb.log("Valid", **{"best_loss": best_metric_score})
            print(f"saved {ckpt_fpath}")


def main(args):
    run_id = make_run_id()
    cur_path = f'{args.model_dir}/{run_id}'
    make_dir(cur_path)
    save_args_to_json(args, cur_path)
    wandb = Wandb(run_id, **args.__dict__)
    wandb.init_wandb()
    
    do_training(wandb, cur_path, **args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
