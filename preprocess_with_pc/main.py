import argparse
import os
import plyfile
import numpy as np
import random
import pickle
from plyfile import PlyData

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--input', type=str, default='../data/demo/pc/597cb92a5bfb580eed98cca8f0ccd5f7.ply', help='input ply file')
    parser.add_argument('--output', type=str, default='../data/demo/597cb92a5bfb580eed98cca8f0ccd5f7.p', help='output pickle file')
    parser.add_argument('--log_dir', type=str, default='log_d60ec6', help='log dir of intermediate files')
    parser.add_argument('--K',  default=50, type=int, help='number of nearest neighbor when proposing candidates')
    return parser.parse_args()

def output_pc_txt(out_file, plydata):
    try:
        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']
    except:
        print("Error: fail to parse the input file.")
    n = len(x)
    if n < 12000 or n > 12800:
        print("Error: The size of point cloud should be between 12,000 ~ 12,800 to fit the pre-trained model.")
        exit()
    with open(out_file, 'w') as f:
        f.write("%d\n" % (n))
        for i in range(n):
            f.write("%f %f %f\n" % (x[i], y[i], z[i]))

def resample(pc, n):
    idx = np.arange(pc.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pc.shape[0], size = n - pc.shape[0])])
    return pc[idx[:n]]

def gen_pickle_file(candidates_txt, pickle_file):
    with open(candidates_txt, 'r') as f:
        lines = f.readlines()
        n = int(lines[0])
        m = int(lines[n + 1])
        pc = np.zeros((n, 3))
        vertex_idx = np.zeros((m, 3), dtype = np.int16)
        label = np.ones(m, dtype = np.int8) * -1

        for j in range(n):
            pc[j][0] = float(lines[j + 1].split()[0])
            pc[j][1] = float(lines[j + 1].split()[1])
            pc[j][2] = float(lines[j + 1].split()[2])

        for j in range(m):
            vertex_idx[j][0] = float(lines[j + 2 + n].split()[0])
            vertex_idx[j][1] = float(lines[j + 2 + n].split()[1])
            vertex_idx[j][2] = float(lines[j + 2 + n].split()[2])
        
    pc = resample(pc, 12800)    
    gt = {'pc': pc, 'vertex_idx': vertex_idx, 'label': label}
    pickle.dump(gt, open(pickle_file,'wb'))


if __name__ == '__main__':
    args = parse_args()
    args.log_dir = "log/%s" % os.path.basename(args.input)[:6]
    os.makedirs(args.log_dir, exist_ok = True) 

    print("Loading input mesh!")
    plydata = PlyData.read(args.input)
    pc_txt = os.path.join(args.log_dir, "pc.txt")
    output_pc_txt(pc_txt, plydata)

    print("Proposing candidates!")
    candidates_txt = os.path.join(args.log_dir, "candidates.txt")
    os.system("./build/propose_candidates %s %s %d"%(pc_txt, candidates_txt, args.K))

    print("Generating pickle file!")
    pickle_file = args.output
    gen_pickle_file(candidates_txt, pickle_file)
    print("Finish!")
