# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import torch
import torch.nn as nn


class BufferList(nn.Module):

    def __init__(self, buffers):
        super(BufferList, self).__init__()
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(i), buffer, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class PointGenerator(nn.Module):

    def __init__(self, strides, buffer_size, offset=False):
        super(PointGenerator, self).__init__()

        reg_range, last = [], 0
        for stride in strides[1:]:
            reg_range.append((last, stride))
            last = stride
        reg_range.append((last, float('inf')))

        self.strides = strides
        self.reg_range = reg_range
        self.buffer_size = buffer_size
        self.offset = offset

        self.buffer = self._cache_points()

    def _cache_points(self):
        buffer_list = []
        for stride, reg_range in zip(self.strides, self.reg_range):
            reg_range = torch.Tensor([reg_range])
            lv_stride = torch.Tensor([stride])
            points = torch.arange(0, self.buffer_size, stride)[:, None]
            if self.offset:
                points += 0.5 * stride
            reg_range = reg_range.repeat(points.size(0), 1)
            lv_stride = lv_stride.repeat(points.size(0), 1)
            buffer_list.append(torch.cat((points, reg_range, lv_stride), dim=1))
        buffer = BufferList(buffer_list)
        return buffer

    def forward(self, pymid):
        points = [] # 初始化一个列表，用于存储最终输出的点集
        sizes = [p.size(1) for p in pymid] + [0] * (len(self.buffer) - len(pymid)) # 计算所有层的点集大小，如果给定的层级在 pymid 中不存在，则大小设置为0
        # 遍历每个层级的实际点集大小以及预先缓存的点集
        for size, buffer in zip(sizes, self.buffer):
            if size == 0:
                continue
            assert size <= buffer.size(0), 'reached max buffer size' # 检查层级的实际点集大小是否超过了预先缓存的最大大小
            points.append(buffer[:size, :]) # 从预先缓存的点集中取出实际需要的点集大小的部分
        points = torch.cat(points)
        return points
