import networks
import torch
import os
import argparse
from main import get_data, get_test_loader
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation code')
    parser.add_argument('--data-dir', default='shanghaitech_part_A',
                        help='directory to test data')
    parser.add_argument('--model-dir', default='./logs/SHA.pth',
                        help='directory to saved model')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    net = networks.create('memMeta')
    checkpoint = torch.load(args.model_dir, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint, strict=False)
    net.cuda()
    net.eval()
    print('=' * 50)
    val_loss = []
    mae = 0.0
    mse = 0.0
    
    test_set = get_data(args.data_dir, source=False)
    test_loader = get_test_loader(test_set, 1, 4)
    for vi, data in enumerate(test_loader, 0):
        img, gt_map = data
        # pdb.set_trace()
        with torch.no_grad():
            img = img.cuda()
            gt_map = gt_map.cuda()
    
            pred_map = net(img)
    
            pred_map = pred_map.data.cpu().numpy()
            gt_map = gt_map.data.cpu().numpy()
    
            gt_count = np.sum(gt_map)/1000.
            pred_cnt = np.sum(pred_map)/1000.
    
            mae += abs(gt_count - pred_cnt)
            mse += ((gt_count - pred_cnt) * (gt_count - pred_cnt))
    
    mae = mae / len(test_loader)
    mse = np.sqrt(mse / len(test_loader))
    print('mae:', mae, 'mse:', mse)
