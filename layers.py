import math
import torch
import torch.nn as nn
import numpy as np


class BatchRotator(object):

    def __init__(self, rotations=360):
        self.rotations = rotations
        self.rot_mat = torch.zeros((rotations, 2, 3))
        self.__init_rot_mat()

    def __call__(self, x, thetas=None):
        if thetas is None:
            rot_mat = self.rot_mat
        else:
            rot_mat = self.rot_mat[thetas]
        grid = torch.nn.functional.affine_grid(rot_mat, x.size(), align_corners=False).to(x.device)
        x = torch.nn.functional.grid_sample(x, grid, align_corners=False)
        return x

    def __init_rot_mat(self):
        for i, theta in enumerate(range(0, 360, int(360//self.rotations))):
            theta = torch.tensor(theta * np.pi / 180)
            self.rot_mat[i] = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                                            [torch.sin(theta), torch.cos(theta), 0]])


class ConvEq2d(nn.Module):

    def __init__(self, in_channels=1, out_channels=2, kernel_size=11, groups=1, padding=0):
        super(ConvEq2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rotations = 8
        self.groups = groups
        self.padding = padding
        self.kernel_size = kernel_size
        self.n_filters = int(in_channels * out_channels // (8 * groups))
        self.rot_shift = 360 // self.rotations
        self.kernel = nn.Parameter((torch.rand(1, self.n_filters, kernel_size, kernel_size) - 0.5) / 10,
                                   requires_grad=True)
        nn.init.xavier_uniform_(self.kernel, gain=nn.init.calculate_gain('relu'))
        self.mask = self.__circular_mask()
        self.rotator = BatchRotator(self.rotations)

    def forward(self, x):
        self.__update_kernels()
        out = torch.nn.functional.conv2d(x, self.kernels.reshape(self.out_channels, self.in_channels // self.groups,
                                                                 self.kernel_size, self.kernel_size).to(x.device),
                                         groups=self.groups, bias=None, padding=self.padding)
        return out

    def __update_kernels(self):
        self.kernels = self.kernel.repeat((self.rotations, 1, 1, 1))
        self.kernels = self.rotator(self.kernels)
        self.kernels = self.kernels.permute(1, 0, 2, 3)
        self.kernels[~self.mask] = 0

    def __circular_mask(self, center=None, radius=None):
        if center is None:
            center = (int(self.kernel_size / 2), int(self.kernel_size / 2))
        if radius is None:
            radius = min(center[0], center[1], self.kernel_size - center[0], self.kernel_size - center[1])

        y, x = np.ogrid[:self.kernel_size, :self.kernel_size]
        dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        mask = dist_from_center <= radius
        mask = mask[None].repeat(self.n_filters, axis=0)[:, None, :, :]
        mask = mask[None].repeat(self.rotations, axis=2)
        return mask

    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, ' \
               f'groups={self.groups}, rotations={self.rotations}'


class Mandala2d(nn.Module):

    def __init__(self, in_size, width, stride):
        super(Mandala2d, self).__init__()
        self.size = in_size
        self.width = width
        self.stride = stride
        self.indices = self.__indices()

    def forward(self, x):
        xm = torch.zeros_like(x)
        for j in range(len(self.indices)):
            v = x[:, :, self.indices[j][:, 0], self.indices[j][:, 1]]
            vi = torch.zeros_like(v)
            a = torch.argmax(v, dim=2)
            # TODO: get rid of this loop
            for i in range(v.shape[0]):
                vi[i] = torch.roll(v[i], shifts=-a[i, 0].item(), dims=1)
            xm[:, :, self.indices[j][:, 0], self.indices[j][:, 1]] = vi

        return xm

    def __indices(self):
        c = self.size / 2 - 0.5, self.size / 2 - 0.5
        md = np.round(np.sqrt(2) * self.size / 2, 0).astype(int)
        nrings = math.ceil(md / self.stride)
        rings = [[] for _ in range(nrings)]
        for i in range(self.size):
            for j in range(self.size):
                d = math.sqrt((c[0] - i) ** 2 + (c[1] - j) ** 2)
                angle = 180 * math.atan2(j - c[1], i - c[0]) / np.pi + 180
                for k, r in enumerate(range(0, md, self.stride)):
                    if r + self.width > d >= r:
                        rings[k].append([i, j, d, angle])
        rings = [np.array(r)[np.array(r)[:, 3].argsort()][:, :2].astype(int) for r in rings if len(r) > 0]
        return rings

    def extra_repr(self):
        return f'in_size={self.size}, width={self.width}, stride={self.stride}'


class CircleMask(object):

    def __init__(self, size, channels):
        self.size = size
        self.channels = channels
        self.mask = self.__circular_mask()

    def __call__(self, sample):
        # print(self.mask.shape)
        # print(sample.shape)
        sample[~self.mask] = 0
        return sample

    def __circular_mask(self, center=None, radius=None):
        if center is None:
            center = (int(self.size / 2), int(self.size / 2))
        if radius is None:
            radius = min(center[0], center[1], self.size - center[0], self.size - center[1])

        y, x = np.ogrid[:self.size, :self.size]
        dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        mask = dist_from_center <= radius
        mask = mask[None].repeat(self.channels, axis=0)[None]
        return mask
