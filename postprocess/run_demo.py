import os
with open("../data/demo/models.txt", "r") as f:
    models = [line.rstrip() for line in f.readlines()]

os.makedirs("../log/shapenet_pretrained/test_demo/output_mesh/", exist_ok = True) 

for model in models:
    pickle_file = "../data/demo/%s.p" % model
    output_file = "../log/shapenet_pretrained/test_demo/output_mesh/%s.ply" % model
    os.system("python3 main.py --pickle_file %s --output %s"%(pickle_file, output_file))