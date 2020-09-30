import argparse
import os
from dataset import TestDataset
import torch
from pathlib import Path
import importlib
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--log_dir', type=str, default='shapenet_pretrained', help='Log path [default: None]')
    parser.add_argument('--npoint', type=int,  default=12800, help='Size of point cloud [default: 12800]')
    parser.add_argument('--ntriangle', type=int,  default=300000, help='Size of candidates per pass per shape')
    parser.add_argument('--output_dir', type=str, default='test_demo', help='dir to store predicted label')
    parser.add_argument('--input_dir', type=str, default='../data/demo', help='dir of input pickle file')
    return parser.parse_args()

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    log_dir = Path('../log/%s' % args.log_dir)
    checkpoints_dir = log_dir.joinpath('checkpoints/')
    output_dir = log_dir.joinpath('%s'%args.output_dir)
    output_dir.mkdir(exist_ok = True)

    Dataset = TestDataset(root = args.input_dir, npoints = args.npoint, ntriangles = args.ntriangle)
    DataLoader = torch.utils.data.DataLoader(Dataset, batch_size = args.batch_size, shuffle = False, num_workers = 12)
    print("The number of test data is: %d." %  len(Dataset))

    MODEL = importlib.import_module("network")
    classifier = MODEL.get_model().cuda()
    classifier = torch.nn.DataParallel(classifier)
    
    checkpoint = torch.load(checkpoints_dir.joinpath('best_model.pth'))
    classifier.load_state_dict(checkpoint['model_state_dict'])
    print('Pretrained model loaded.')

    with torch.no_grad():
        total_correct = 0
        total_seen = 0
        classifier = classifier.eval()

        for batch_id, (pc, vertex_idx, label, model_ids) in tqdm(enumerate(DataLoader), total = len(DataLoader), smoothing = 0.9):
            B, _, _ = pc.size()
            pc, vertex_idx = pc.float().cuda(), vertex_idx.long().cuda()
            pred = classifier(pc, vertex_idx)
            pred = pred.contiguous().view(-1, 3).max(1)[1]
            pred = pred.cpu().data.numpy()
            label = label.view(-1).data.numpy()
            total_correct += np.sum(pred == label)
            total_seen += (B * args.ntriangle)
            pred = pred.reshape((B, args.ntriangle))

            for i in range(B):
                np.save(os.path.join(output_dir, '%s.npy' % model_ids[i]), pred[i])
            
        print(total_correct / float(total_seen))

if __name__ == '__main__':
    args = parse_args()
    main(args)