import argparse
import os
from dataset import TrainDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import numpy as np
from utils import weights_init, bn_momentum_adjust

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch',  default=50, type=int, help='Epoch to run [default: 50]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--dataset_dir', type=str, default="../data/all/", help='Dataset path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int,  default=12800, help='Point Number [default: 12800]')
    parser.add_argument('--ntriangle', type=int,  default=25000, help='Triangle Number [default: 100000]')
    parser.add_argument('--step_size', type=int,  default=10, help='Decay step for lr decay [default: every 20 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.8, help='Decay rate for lr decay [default: 0.5]')
    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    timestr = str(datetime.datetime.now().strftime('%m-%d_%H-%M'))
    experiment_dir = Path('../log/')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    shutil.copy('dataset.py', str(experiment_dir))
    shutil.copy('network.py', str(experiment_dir))
    shutil.copy('train.py', str(experiment_dir))
    shutil.copy('utils.py', str(experiment_dir))

    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    
    TRAIN_DATASET = TrainDataset(root = args.dataset_dir, npoints = args.npoint, ntriangles = args.ntriangle, split = 'train')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size = args.batch_size, shuffle = True, num_workers = 12)
    VAL_DATASET = TrainDataset(root = args.dataset_dir, npoints = args.npoint, ntriangles = args.ntriangle, split = 'val')
    valDataLoader = torch.utils.data.DataLoader(VAL_DATASET, batch_size = args.batch_size, shuffle = False, num_workers = 12)
    log_string("The number of train data is: %d" % len(TRAIN_DATASET))
    log_string("The number of val data is: %d" %  len(VAL_DATASET))

    MODEL = importlib.import_module("network")
    classifier = MODEL.get_model()
    criterion = MODEL.get_loss()
    classifier = torch.nn.DataParallel(classifier).cuda()
    criterion = torch.nn.DataParallel(criterion).cuda()

    try:
        checkpoint = torch.load(checkpoints_dir.joinpath('best_model.pth'))
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['test_acc']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Pretrained model loaded...')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        best_acc = 0
        classifier = classifier.apply(weights_init)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )


    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0

    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
       
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        log_string('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x,momentum))
      
        classifier = classifier.train()

        acc_buffer = []
        loss_2_buffer = []
        loss_3_buffer = []
        loss_buffer = []

        for i, (pc, vertex_idx, label) in tqdm(enumerate(trainDataLoader), total = len(trainDataLoader), smoothing = 0.9):
            pc, vertex_idx, label = pc.cuda(), vertex_idx.cuda(), label.cuda()
            B, _, _ = pc.size()
            optimizer.zero_grad()

            pred = classifier(pc, vertex_idx)
            pred = pred.contiguous().view(-1, 3)
            label = label.view(-1)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(label.data).sum().cpu()
            acc_buffer.append(correct.item() / (B * args.ntriangle))
            loss_2, loss_3 = criterion(pred, label)
            loss = (loss_2 + loss_3).mean()
            loss.backward()
            optimizer.step()
            loss_2_buffer.append(loss_2.mean().cpu().item())
            loss_3_buffer.append(loss_3.mean().cpu().item())
            loss_buffer.append(loss.cpu().item())
        
        train_acc = np.mean(acc_buffer)    
        log_string('Train accuracy: %.5f' % train_acc)
        log_string('Train loss: %.5f' % np.mean(loss_buffer))
        log_string('Train loss_2: %.5f' % np.mean(loss_2_buffer))
        log_string('Train loss_3: %.5f' % np.mean(loss_3_buffer))
       

        with torch.no_grad():
            classifier = classifier.eval()
            acc_buffer = []
            loss_2_buffer = []
            loss_3_buffer = []
            loss_buffer = []

            for i, (pc, vertex_idx, label) in tqdm(enumerate(valDataLoader), total = len(valDataLoader), smoothing = 0.9):
                pc, vertex_idx, label = pc.cuda(), vertex_idx.cuda(), label.cuda()
                B, _, _ = pc.size()
                pred = classifier(pc, vertex_idx)
                pred = pred.contiguous().view(-1, 3)
                label = label.view(-1)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(label.data).sum().cpu()
                acc_buffer.append(correct.item() / (B * args.ntriangle))
                loss_2, loss_3 = criterion(pred, label)
                loss = (loss_2 + loss_3).mean()
                loss_2_buffer.append(loss_2.mean().cpu().item())
                loss_3_buffer.append(loss_3.mean().cpu().item())
                loss_buffer.append(loss.cpu().item())
                
            test_acc = np.mean(acc_buffer)
            log_string('Val accuracy: %.5f' % test_acc)
            log_string('Val loss: %.5f' % np.mean(loss_buffer))
            log_string('Val loss_2: %.5f' % np.mean(loss_2_buffer))
            log_string('Val loss_3: %.5f' % np.mean(loss_3_buffer))

            
        if (test_acc >= best_acc):
            best_acc = test_acc
            log_string('Saving model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            state = {
                'epoch': epoch,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            
        log_string('Saving model....')
        savepath = str(checkpoints_dir) + '/last_model.pth'
        state = {
            'epoch': epoch,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)

        log_string('Best accuracy is: %.5f'%best_acc)
        global_epoch += 1

if __name__ == '__main__':
    args = parse_args()
    main(args)