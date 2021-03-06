import torch.nn as nn
import torch.nn.functional as functional

from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling
from layers import ConvEq2d, Mandala2d


class ModelZ2CNN(nn.Module):
    """
    Testing taco model described in their paper. This is the 'base' model without
    equivariant layers.
    https://tacocohen.files.wordpress.com/2016/06/gcnn.pdf
    """
    def __init__(self):
        super(ModelZ2CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=80, kernel_size=(3, 3))
        self.cnn2 = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=(3, 3))
        self.max1 = nn.MaxPool2d(kernel_size=2)
        self.norm = nn.BatchNorm2d(80)
        self.cnn3 = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=(3, 3))
        self.cnn4 = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=(3, 3))
        self.cnn5 = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=(3, 3))
        self.cnn6 = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=(3, 3))
        self.cnn7 = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=(4, 4))
        self.fc = nn.Linear(80, 10)

    def forward(self, x):
        x = functional.relu(self.cnn1(x))
        x = functional.relu(self.cnn2(x))
        x = self.norm(x)
        x = functional.dropout(x)
        x = self.max1(x)
        x = functional.relu(self.cnn3(x))
        x = functional.relu(self.cnn4(x))
        x = functional.relu(self.cnn5(x))
        x = functional.relu(self.cnn6(x))
        x = functional.relu(self.cnn7(x))
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class ModelP4CNNP4(nn.Module):
    """
    Testing taco model architecture with Global invariance. This is the model with
    equivariant layers. As this model uses equivariance layers and ends up reducing the feature
    map to a 1x1 map it will also have global invariance.
    """
    def __init__(self):
        super(ModelP4CNNP4, self).__init__()
        self.cnn1 = P4ConvZ2(1, 20, kernel_size=3)
        self.cnn2 = P4ConvP4(20, 20, kernel_size=3)
        self.cnn3 = P4ConvP4(20, 20, kernel_size=3)
        self.cnn4 = P4ConvP4(20, 20, kernel_size=3)
        self.cnn5 = P4ConvP4(20, 20, kernel_size=3)
        self.cnn6 = P4ConvP4(20, 20, kernel_size=3)
        self.cnn7 = P4ConvP4(20, 20, kernel_size=4)
        self.fc = nn.Linear(80, 10)

    def forward(self, x):
        x = functional.relu(self.cnn1(x))
        x = functional.relu(self.cnn2(x))
        # x = functional.batch_norm(x)
        x = functional.dropout(x)
        x = plane_group_spatial_max_pooling(x, 2, 2)
        x = functional.relu(self.cnn3(x))
        x = functional.relu(self.cnn4(x))
        x = functional.relu(self.cnn5(x))
        x = functional.relu(self.cnn6(x))
        x = functional.relu(self.cnn7(x))
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class ModelConvEq2D(nn.Module):
    # TODO: Gif bestia con todas las capas
    # TODO: Y las capas de dos en dos con groups = 40. Fuera cnn7 y avgpool 4
    def __init__(self):
        super(ModelConvEq2D, self).__init__()
        self.cnn1 = ConvEq2d(in_channels=1, out_channels=80, kernel_size=3)
        self.cnn2 = ConvEq2d(in_channels=80, out_channels=80, kernel_size=3)
        self.max1 = nn.MaxPool2d(kernel_size=2)
        # self.norm = nn.BatchNorm2d(40)
        self.cnn3 = ConvEq2d(in_channels=80, out_channels=80, kernel_size=3)
        self.cnn4 = ConvEq2d(in_channels=80, out_channels=80, kernel_size=3)
        self.cnn5 = ConvEq2d(in_channels=80, out_channels=80, kernel_size=3)
        self.cnn6 = ConvEq2d(in_channels=80, out_channels=80, kernel_size=3)
        self.cnn7 = ConvEq2d(in_channels=80, out_channels=80, kernel_size=4)
        self.fc = nn.Linear(80, 10)

    def forward(self, x):
        x = functional.relu(self.cnn1(x))
        x = functional.relu(self.cnn2(x))
        x = functional.dropout(x)
        # x = self.norm(x)
        x = functional.relu(self.max1(x))
        x = functional.relu(self.cnn3(x))
        x = functional.relu(self.cnn4(x))
        x = functional.relu(self.cnn5(x))
        x = functional.relu(self.cnn6(x))
        x = functional.relu(self.cnn7(x))
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
