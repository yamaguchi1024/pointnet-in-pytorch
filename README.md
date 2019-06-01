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
