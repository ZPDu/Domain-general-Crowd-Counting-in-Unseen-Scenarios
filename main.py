from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import random
import numpy as np
import time
from datetime import timedelta
import networks
from utils.data import IterLoader
from utils.data.preprocessor import Preprocessor
from utils.data.preprocessor_tran import Preprocessor_tran
from utils.data import transforms as T
import datasets
from utils.trainer_vanilla import Trainer
from evaluator import Evaluator
from utils.clustering.domain_split import domain_split

start_epoch = best_mAP = 0

def get_data(data_dir, source, num_domains=None):
    if source:
        root = osp.join(data_dir, 'train_data')
        dataset = datasets.create('CrowdCluster', root, num_domains)
    else:
        root = osp.join(data_dir, 'test_data')
        dataset = datasets.create('Crowd', root)
    return dataset

def get_train_loader(args, dataset, height, width, batch_size, workers,
                    iters):

    normalizer = T.standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.RandomHorizontallyFlip(),
             T.RandomCrop((height,width))])
    img_transformer = T.standard_transforms.Compose([
        T.standard_transforms.ToTensor(),
        normalizer
    ])
    gt_transformeer = T.standard_transforms.Compose([
        T.LabelNormalize(1000.)
    ])


    train_set = sorted(dataset.train)

    train_loader = IterLoader(
                DataLoader(Preprocessor_tran(train_set, root=dataset.root, main_transform=train_transformer,
                            img_transform=img_transformer, gt_transform=gt_transformeer),
                            batch_size=batch_size, num_workers=workers, sampler=None,
                            shuffle=True, pin_memory=False, drop_last=True), length=None)

    return train_loader

def get_test_loader(dataset, batch_size, workers):
    normalizer = T.standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    test_transformer = None
    img_transformer = T.standard_transforms.Compose([
        T.standard_transforms.ToTensor(),
        normalizer
    ])
    gt_transformer = T.standard_transforms.Compose([
        T.LabelNormalize(1000.)
    ])

    testset = dataset

    test_loader = DataLoader(
        Preprocessor(testset.train, root=dataset.root, main_transform=test_transformer, img_transform=img_transformer, gt_transform=gt_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=False)

    return test_loader

def create_model(args):
    model = networks.create(args.arch)
    model.cuda()
    optim = None
    if args.resume:
        global best_mae, best_mse, start_epoch, optim_dict
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        best_mae = checkpoint['mae']
        best_mse = checkpoint['mse']
        start_epoch = checkpoint['epoch']
        optim = checkpoint['optim']
    return  model, optim

def online_clustering(source, model, epoch, num_clustering):
    device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
    pseudo_domain_label = domain_split(source, model, device=device,
                                       cluster_before=source.clusters,
                                       filename=None, epoch=epoch,
                                       nmb_cluster=num_clustering, method='Kmeans',
                                       pca_dim=256, whitening=False, L2norm=False, instance_stat=True)
    source.set_cluster(np.array(pseudo_domain_label))

def main_worker(args):
    global best_mae, best_mse, start_epoch
    best_mae = 100000
    best_mse = 100000
    start_time = time.monotonic()
    start_epoch = 0
    optim_dict = None

    cudnn.benchmark = True
    iters = args.iters if (args.iters > 0) else None

    #Create Model
    model, optim = create_model(args)

    #Prepare data
    print("==> Load datasets")

    dataset_src = get_data(args.data_dir, True, args.num_clustering)
    dataset_trg = get_data(args.data_dir, False)

    test_loader = get_test_loader(dataset_trg, args.test_batch_size, args.workers)

    #Evaluator
    evaluator = Evaluator(model)
    # if args.evaluate:
    #     evaluator.evaluate(test_loader)
    #     return

    #Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if optim is not None:
        optimizer.load_state_dict(optim)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    criterion = nn.MSELoss(reduction='sum').cuda()
    trainer = Trainer(args, model, criterion)

    for epoch in range(start_epoch, args.epochs):
        print('==> start training epoch {} \t ==> learning rate = {}'.format(epoch, optimizer.param_groups[0]['lr']))
        torch.cuda.empty_cache()
        #Online clustering
        if epoch % args.cluster_step == 0:
            online_clustering(dataset_src, model, epoch, args.num_clustering)
            datasets_src, train_loaders = [], []
            for src in dataset_src.subdomains:
                train_loader = get_train_loader(args, src, args.height, args.width,
                                                args.batch_size, args.workers, iters)
                train_loaders.append(train_loader)

        # start training
        trainer.train(epoch, train_loaders, optimizer,
                      print_freq=args.print_freq, train_iters=args.iters)
        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mae, mse = evaluator.evaluate(test_loader)
            is_best = (mae < best_mae)

            # save model
            saved_model = {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'mae': best_mae,
                'mse': best_mse,
                'optim': optimizer.state_dict()
            }
            if is_best:
                best_mae = mae
                best_mse = mse
                saved_model['mae'] = best_mae
                saved_model['mse'] = best_mse
                torch.save(saved_model, osp.join(args.logs_dir, 'bestmodel.pth.tar'))
            torch.save(saved_model, osp.join(args.logs_dir, 'latestmodel.pth.tar'))


            print('\n * Finished epoch {:3d}  model mae: {:5.1f} mse: {:5.1f}  best: {:5.1f}{}\n'.
                  format(epoch, mae, mse, best_mae, ' *' if is_best else ''))

    print('==> Test with the best model:')
    checkpoint = torch.load(osp.join(args.logs_dir, 'bestmodel.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader)
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training code for Domain-general Crowd Counting in Unseen Scenarios")
    # data
    parser.add_argument('--num-clustering', type=int, default=4)
    parser.add_argument('--cluster-step', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('--test-batch-size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=0)
    parser.add_argument('--height', type=int, default=320, help="input height")
    parser.add_argument('--width', type=int, default=320, help="input width")

    # model
    parser.add_argument('-a', '--arch', type=str, default='msMeta',
                        choices=networks.names())
    # optimizer
    parser.add_argument('--lr', type=float, default=1e-5,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=135)
    parser.add_argument('--iters', type=int, default=100)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=25)
    parser.add_argument('--eval-step', type=int, default=1)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(''))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join('', 'logs'))
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    main()
