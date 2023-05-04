# Pytorch学习

# 完成的内容

## DataSet

DataSet主要用于数据读取。最主要的两个函数实现为

```python
__Len__  __getitem__
__init__：可以在这里设置加载的data和label
__Len__：获取数据集大小
__getitem__：根据索引获取一条训练的数据和标签
```

## DataLoader

DataLoader是Pytorch中用来处理模型输入数据的一个工具类。

常见的参数设置：

+ batch_size(int, optional) : 每个batch有多少个样本，即每次加载几个样本。
+ shuffle(bool, optional) : 在每次开始加载数据集的时候，对数据是否进行重新排序。默认情况下为False。
+ sampler(Sampler, optional): 自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False。默认为随机抓取。
+  num_workers (int, optional) : 加载数据时采用几个进程。默认为0，表示为采用一个主进程加载数据。
+ drop_last (bool, optional) : 是否对batch_size之外的样本进行丢弃。如数据集大小为100，batch_size为3，则会留下1个样本在外。如果设置为False，那么这1个样本会接着加载，只是最后的batch_size会小一点；如果设置为True，则这1个样本将被丢弃

以下为一个实例：

数据集CIFAR10已提前下载好

```python
test_data = torchvision.datasets.CIFAR10("./datasets", train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
writer = SummaryWriter("dataloader")
step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images("test_data_1",imgs,step)
    step = step+1
writer.close()
```

第一次设置batch_size=4,第二次设置batch_size=64



