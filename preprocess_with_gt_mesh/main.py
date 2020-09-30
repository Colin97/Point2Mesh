import argparse
import os
import plyfile
import numpy as np
import random
import pickle
from plyfile import PlyData

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--input', type=str, default='../data/demo/gt_mesh/597cb92a5bfb580eed98cca8f0ccd5f7.ply', help='input ply file')
    parser.add_argument('--output', type=str, default='../data/demo/597cb92a5bfb580eed98cca8f0ccd5f7.p', help='output pickle file')
    parser.add_argument('--log_dir', type=str, default='log_d60ec6', help='log dir of intermediate files')
    parser.add_argument('--K',  default=50, type=int, help='number of nearest neighbor when proposing candidates')
    parser.add_argument('--tau', default=1.3, type=float, help='threshold used to filter out incorrect candidates')
    return parser.parse_args()

def output_mesh_txt(out_file, plydata):
    try:
        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']
        indices = plydata['face']['vertex_indices']    
    except:
        print("Error: fail to parse the input file.")
    n = len(x)
    m = len(indices)
    if n > 100000 or m > 500000:
        print("Error: Too large mesh file.")
        exit()
    if n == 0 or m == 0:
        print("Error: There should contain at least one vertex and one face.")
        exit()
    with open(out_file, 'w') as f:
        f.write("%d\n%d\n" % (n, m))
        for i in range(n):
            f.write("%f %f %f\n" % (x[i], y[i], z[i]))
        for i in range(m):
            f.write("%d %d %d\n" % (indices[i][0], indices[i][1], indices[i][2]))

def resample(pc, n):
    idx = np.arange(pc.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pc.shape[0], size = n - pc.shape[0])])
    return pc[idx[:n]]

def gen_pickle_file(label_txt, pc_txt, pickle_file):
    with open(label_txt, 'r') as f:
        lines = f.readlines()
    n = int(lines[0])
    vertices = np.zeros((n, 3), dtype = np.int16)
    flag = np.zeros(n, dtype = np.int8)
    for j in range(n):
        vertices[j][0] = int(lines[j + 1].split()[0])
        vertices[j][1] = int(lines[j + 1].split()[1])
        vertices[j][2] = int(lines[j + 1].split()[2])
        flag[j] = int(lines[j + 1].split()[3])
        
    with open(pc_txt, 'r') as f:
        lines = f.readlines()
    n = int(lines[0])
    pc = np.zeros((n, 3))
    for j in range(n):
        pc[j][0] = float(lines[j + 1].split()[0])
        pc[j][1] = float(lines[j + 1].split()[1])
        pc[j][2] = float(lines[j + 1].split()[2])
    pc = resample(pc, 12800)    
    gt = {'pc': pc, 'vertex_idx': vertices, 'label': flag}
    pickle.dump(gt, open(pickle_file,'wb'))


if __name__ == '__main__':
    args = parse_args()
    args.log_dir = "log/%s" % os.path.basename(args.input)[:6]
    os.makedirs(args.log_dir, exist_ok = True) 

    print("Loading input mesh!")
    plydata = PlyData.read(args.input)
    mesh_txt = os.path.join(args.log_dir, "mesh.txt")
    output_mesh_txt(mesh_txt, plydata)

    print("Preprocessing input mesh!")
    new_mesh_txt = os.path.join(args.log_dir, "new_mesh.txt")
    os.system("./build/preprocess_mesh %s %s"%(mesh_txt, new_mesh_txt))

    print("Sampling point cloud (12000~12800 points, using binary search to determine radius)!")
    pc_txt = os.path.join(args.log_dir, "pc.txt")
    os.system("./build/sample_pc %s %s 12000 12800"%(new_mesh_txt, pc_txt))

    print("Calculating geodesic distances (may take up to 1 minute)!")
    geo_dis_txt = os.path.join(args.log_dir, "geo_dis.txt")
    os.system("./build/calc_geo_dis %s %s %s"%(pc_txt, new_mesh_txt, geo_dis_txt))

    print("Proposing candidates and calculating distances to the gt mesh!")
    candidates_txt = os.path.join(args.log_dir, "candidates.txt")
    os.system("./build/propose_candidates %s %s %s %d"%(pc_txt, new_mesh_txt, candidates_txt, args.K))

    print("Calculating candidates' label!")
    label_txt = os.path.join(args.log_dir, "label.txt")
    os.system("./build/calc_candidates_label %s %s %s %s %f"%(pc_txt, geo_dis_txt, candidates_txt, label_txt, args.tau))

    print("Generating pickle file!")
    pickle_file = args.output
    gen_pickle_file(label_txt, pc_txt, pickle_file)
    print("Finish!")
