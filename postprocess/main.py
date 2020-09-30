import argparse
import os
import plyfile
import numpy as np
import random
import pickle

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--pickle_file', type=str, default='../data/demo/597cb92a5bfb580eed98cca8f0ccd5f7.p', help='preprocessed pickle file')
    parser.add_argument('--pred_dir', type=str, default='../log/shapenet_pretrained/test_demo', help='dir of predicted labels')
    parser.add_argument('--output', type=str, default='../log/shapenet_pretrained/test_demo/output_mesh/597cb92a5bfb580eed98cca8f0ccd5f7.ply', help='output ply file')
    parser.add_argument('--log_dir', type=str, default=None, help='log dir of intermediate files')
    parser.add_argument('--ntriangle', type=int,  default=300000, help='Size of candidates per pass per shape')
    return parser.parse_args()

def merge(args, model_id, pred_txt):
    preprocessed_data = pickle.load(open(args.pickle_file, 'rb'))
    pc = preprocessed_data['pc']
    idx = preprocessed_data['vertex_idx']
    n = pc.shape[0]
    m = idx.shape[0]

    pred = []
    for i in range((m + args.ntriangle - 1) // args.ntriangle):
        pred.append(np.load(os.path.join(args.pred_dir,  '%s_%d.npy'%(model_id, i))))
    pred = np.concatenate(pred, axis = 0)
    pred = pred[: m]
        
    with open(pred_txt, 'w') as f:
        f.write('%d\n' % n)
        for i in range(n):
            f.write('%f %f %f\n' % (pc[i][0], pc[i][1], pc[i][2]))
        f.write('%d\n' % m)
        for i in range(m):
            f.write("%d %d %d %d\n" % (idx[i][0], idx[i][1], idx[i][2], pred[i]))
        
if __name__ == '__main__':
    args = parse_args()
    model_id = os.path.basename(args.pickle_file).split('.p')[0] 
    args.log_dir = "log/%s" % model_id[:6]
    os.makedirs(args.log_dir, exist_ok = True) 

    print("Merging npy files!")
    pred_txt = os.path.join(args.log_dir, 'pred.txt')
    merge(args, model_id, pred_txt)

    print("Postprocessing!")
    os.system("./build/postprocess %s %s" % (pred_txt, args.output))

    print("Finish!")
