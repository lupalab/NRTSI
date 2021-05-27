from datasets import get_dataset
import pdb
import torch
train_data, test_data = get_dataset('./data', 'billiard', 0)
diff = (test_data[:,1:,:] - test_data[:,:-1, :]).abs()

mean = torch.mean(diff, 1, True)
std = torch.std(diff, 1, keepdim=True)
mask = (diff - mean).abs() < std
speed = torch.sum(mask * diff, 1) / torch.sum(mask, 1) 
speed = (speed ** 2).sum(1).sqrt()
print(speed.min(), speed.max())
print(train_data.max(), train_data.min())