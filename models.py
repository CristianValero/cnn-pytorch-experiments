import torch.nn as nn
import torch.nn.functional as functional
from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling
from layers import ConvEq2d


class ModelZ2CNN(nn.Module):
    """
    Testing taco model described in their paper. This is the 'base' model without
    equivariant layers.
    https://tacocohen.files.wordpress.com/2016/06/gcnn.pdf

    epoch:   1, train_loss: 2.157860, test_acc: 0.5478
    ...
    epoch:  10, train_loss: 0.178814, test_acc: 0.9523
    """
    def __init__(self):
        super(ModelZ2CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(3, 3))
        self.cnn2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3))
        self.max1 = nn.MaxPool2d(kernel_size=2)
        self.norm = nn.BatchNorm2d(20)
        self.cnn3 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3))
        self.cnn4 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3))
        self.cnn5 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3))
        self.cnn6 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(3, 3))
        self.cnn7 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(4, 4))
        self.fc = nn.Linear(20, 10)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.max1(x)
        x = self.norm(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        x = self.cnn7(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class ModelP4CNNP4(nn.Module):
    """
    Testing taco model architecture with Global invariance. This is the 'Equivariant' model with
    equivariant layers. As this model uses equivariance layers and ends up reducing the feature
    map to a 1x1 map it will also have global invariance.

    epoch:   1, train_loss: 0.902755, test_acc: 0.8273
    ...
    epoch:   5, train_loss: 0.081899, test_acc: 0.9719
    """
    def __init__(self):
        super(ModelP4CNNP4, self).__init__()
        self.cnn1 = P4ConvZ2(1, 20, kernel_size=3)
        self.cnn2 = P4ConvP4(20, 20, kernel_size=3)
        self.max1 = nn.MaxPool2d(kernel_size=2)
        self.norm = nn.BatchNorm2d(20)
        self.cnn3 = P4ConvP4(20, 20, kernel_size=3)
        self.cnn4 = P4ConvP4(20, 20, kernel_size=3)
        self.cnn5 = P4ConvP4(20, 20, kernel_size=3)
        self.cnn6 = P4ConvP4(20, 20, kernel_size=3)
        self.cnn7 = P4ConvP4(20, 20, kernel_size=4)
        self.fc = nn.Linear(80, 10)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = plane_group_spatial_max_pooling(x, 2, 2)
        # x = self.max1(x)
        # x = self.norm(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        x = self.cnn7(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class ModelConvEq2D(nn.Module):
    """
    Testing taco model architecture with Global invariance. This is the 'Equivariant' model with
    ConvEq2d layers.

    epoch:   1, train_loss: 0.902755, test_acc: 0.8273
    ...
    epoch:  10, train_loss: 0.194891, test_acc: 0.9204
    """
    def __init__(self):
        super(ModelConvEq2D, self).__init__()
        self.cnn1 = ConvEq2d(1, 20, kernel_size=3)
        self.cnn2 = ConvEq2d(20, 20, kernel_size=3)
        self.max1 = nn.MaxPool2d(kernel_size=2)
        self.norm = nn.BatchNorm2d(20)
        self.cnn3 = ConvEq2d(20, 20, kernel_size=3)
        self.cnn4 = ConvEq2d(20, 20, kernel_size=3)
        self.cnn5 = ConvEq2d(20, 20, kernel_size=3)
        self.cnn6 = ConvEq2d(20, 20, kernel_size=3)
        self.cnn7 = ConvEq2d(20, 20, kernel_size=4)
        self.fc = nn.Linear(80, 10)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.max1(x)
        x = self.norm(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        x = self.cnn7(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
