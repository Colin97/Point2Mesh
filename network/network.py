import torch
import torch.nn as nn
import torch.nn.functional as F
import sparseconvnet as scn

class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        self.part_num = 3
        self.resolution = 150
        self.dimension = 3
        self.reps = 2 #Conv block repetition factor
        self.m = 32 #Unet number of features
        self.nPlanes = [self.m, 2 * self.m, 3 * self.m, 4 * self.m, 5 * self.m] #UNet number of features per level
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(self.dimension, torch.LongTensor([self.resolution * 8 + 15] * 3), mode=3)).add(
           scn.SubmanifoldConvolution(self.dimension, 1, self.m, 3, False)).add(
           scn.FullyConvolutionalNet(self.dimension, self.reps, self.nPlanes, residual_blocks=False, downsample=[3,2])).add(
           scn.BatchNormReLU(sum(self.nPlanes))).add(
           scn.OutputLayer(self.dimension))
        self.nc = 64
        self.linear = nn.Linear(sum(self.nPlanes), self.nc)
        self.convs1 = torch.nn.Conv1d(self.nc * 3, 128, 1)
        self.convs2 = torch.nn.Conv1d(128, 64, 1)
        self.convs3 = torch.nn.Conv1d(64, self.part_num, 1)
        self.bns1 = nn.BatchNorm1d(128)
        self.bns2 = nn.BatchNorm1d(64)

    def forward(self, pc, idx):
        B, N, _ = pc.size()
        _, M, _ = idx.size()

        pc = pc.view(-1, 3)
        pc = (pc * 2 + 4) * self.resolution
        pc = torch.floor(pc).long()
        x = torch.cat((pc, torch.arange(B).unsqueeze(-1).cuda().repeat(1, N).view(-1, 1)), 1)
        x = self.sparseModel([x, torch.ones((B * N, 1)).cuda()])
        x = self.linear(x)
        x = x.view(B, N, self.nc)
        x = x.transpose(1, 2)
       
        x = x.unsqueeze(-1).repeat(1, 1, 1, 3)
        idx = idx.unsqueeze(1).repeat(1, self.nc, 1, 1)
        x = x.gather(2, idx)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(B, self.nc * 3, M)
    
        x = F.relu(self.bns1(self.convs1(x)))
        x = F.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.part_num), dim=-1)
        x = x.view(B, M, self.part_num)

        return x


class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        pred1 = torch.cat([pred[:,0].unsqueeze(-1), torch.max(pred[:,1:], 1, keepdim = True)[0]], 1)
        target1 = target.clone()
        target1[target1 != 0] = 1
        loss_2 = F.nll_loss(pred1, target1)
        loss_3 = F.nll_loss(pred, target)
        return loss_2, loss_3