import networks
import torch
from main_cluster import get_data, get_test_loader
import numpy as np

net = networks.create('memMeta')
checkpoint = torch.load('/home/cbgmm/projects/Zhipeng/DGCount/logs/bestmodel.pth.tar', map_location=torch.device('cpu'))
net.load_state_dict(checkpoint['state_dict'])
net.cuda()
net.eval()
print('=' * 50)
val_loss = []
mae = 0.0
mse = 0.0

# test_set = get_data('SHB', '/home/cbgmm/projects/Zhipeng/ProcessedData/CSHB')
test_set = get_data('QNRF', '/home/cbgmm/projects/Zhipeng/ProcessedData/CQNRF')
# test_set = get_data('SHA', '/home/cbgmm/projects/Zhipeng/ProcessedData/CSHA')
test_loader = get_test_loader(test_set, None, None, 1, 4)
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
