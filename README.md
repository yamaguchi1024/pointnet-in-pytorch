### (Probably) the simplest PointNet implementation in Pytorch
- Neural network uses only 10 lines of code (model.py)
- Operates shape classification task on ShapeNet

### Usage
```
cd pointnet.pytorch
./download.sh
cd ..
python3 classify.py
```

### Requirements
- Pytorch
- CUDA (You can workaround by commenting out something.cuda() calls)

### Classification performance
On [A subset of shapenet](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html), epoch=250, batchsize=32, number of points in the pointcloud data = 2500.
|  | Overall Acc |
| :---: | :---: |
| Original implementation | N/A |
| pointnet.pytorch(w/o feature transform) | 98.1 |
| pointnet.pytorch(w/ feature transform) | 97.7 |
| This Implementation | 99.8 |
