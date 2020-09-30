import os
with open("../data/demo/models.txt", "r") as f:
    models = [line.rstrip() for line in f.readlines()]

for model in models:
    input_file = "../data/demo/pc/%s.ply" % model
    output_file = "../data/demo/%s.p" % model
    os.system("python3 main.py --input %s --output %s"%(input_file, output_file))