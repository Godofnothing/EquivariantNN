import torch

# Here we define discrete transforms

# identity
def identity(x : torch.Tensor):
    return x

# flip
def hflip(x : torch.Tensor):
    return torch.flip(x, dims=[-1])

# rotate by pi * k / 2
def rotate90(x : torch.Tensor):
    return torch.rot90(x, 1, [-2, -1])

def rotate180(x : torch.Tensor):
    return torch.flip(x, dims=[-2, -1])

def rotate270(x : torch.Tensor):
    return torch.rot90(x, -1, [-2, -1])

# rotate + flip
def rotate90_hflip(x : torch.Tensor):
    return hflip(rotate90(x))

def rotate180_hflip(x : torch.Tensor):
    return hflip(rotate180(x))

def rotate270_hflip(x : torch.Tensor):
    return hflip(rotate270(x))