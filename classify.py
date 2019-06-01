import random
import torch
import torch.optim as optim
import torch.utils.data
import sys 
sys.path.insert(0, './pointnet.pytorch')
from dataset import ShapeNetDataset
from model import PointNetCls
import torch.nn.functional as F

# Hyper parameters
# path to shapenet
datapath = '/home/yuka/shapenetcore_partanno_segmentation_benchmark_v0/'
# resample to some points
num_points = 2500
# how many inputs you load to NN at the same time
batchsize = 20
# how many epoch to iterate
epochsize = 1

# load shapenet datasets
dataset = ShapeNetDataset(datapath, classification=True, npoints=num_points)
testset = ShapeNetDataset(datapath, classification=True, npoints=num_points)

dataloader = torch.utils.data.DataLoader(dataset, batchsize, shuffle=True, num_workers=5)
testdataloader = torch.utils.data.DataLoader(testset, batchsize, shuffle=True, num_workers=5)

num_classes = len(dataset.classes)
print(num_classes)

# classifier initialization
classifier = PointNetCls(k=num_classes)

# optimizer
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9,0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

for epoch in range(epochsize):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points = points.cuda()
        target = target.cuda()
        
        optimizer.zero_grad()
        classifier = classifier.train()
        loss = F.nll_loss(classifier.forward(points), target)
        loss.backward()

        optimizer.step()
        print(i)

total_correct = 0
total_testset = 0
classifier = classifier.eval()
for i, data in enumerate(testdataloader):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points = points.cuda()
    target = target.cuda()

    pred = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("accuracy: ", total_correct/total_testset)
