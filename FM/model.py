
import math
import random

import numpy as np

from mindspore import Tensor, nn, context, Parameter, ParameterTuple
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
import mindspore.ops as ops
import mindspore.ops.functional as F
import mindspore.common.dtype as mstype
import mindspore

from initializer import XavierUniform




class FeaturesLinear(nn.Cell):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()

        self.offsets = Tensor([np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)])  # 每个类别的id的起始地址

        self.nochange_conv = nn.Conv2d(1, 1, (1, 1), stride=1)
        self.nochange_conv.weight.set_data(Tensor([[[[1]]]]))
        self.nochange_conv.weight.requires_grad = False
        
        self.embedding = nn.Embedding(int(sum(field_dims)), output_dim)
        self.bias = Parameter(np.zeros((output_dim,), dtype=np.float32))




    def construct(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        每个输入长度39，每个值 当前位置的种类id
        """


        x = x + self.offsets  # 每个位置的种类id在整个embedding中的地址

        x = ops.ExpandDims()(x, 1)
        x = ops.ExpandDims()(x, 1)
        x = self.nochange_conv(x)
        x = x.squeeze()
        x = x.astype(mindspore.int32)  # 作为index int 

        out = self.embedding(x)
        # out = norm(out)
        out = ops.ReduceSum()(out, 1) + self.bias


        return out  


class FeaturesEmbedding(nn.Cell):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(int(sum(field_dims)), embed_dim)
        self.offsets = Tensor(np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32))


        self.nochange_conv = nn.Conv2d(1, 1, (1, 1), stride=1)
        self.nochange_conv.weight.set_data(Tensor([[[[1]]]]))
        self.nochange_conv.weight.requires_grad = False

    def construct(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        x [bs,39]
        self.offsets []
        """

        x = x + self.offsets

        x = ops.ExpandDims()(x, 1)
        x = ops.ExpandDims()(x, 1)
        x = self.nochange_conv(x)
        x = x.squeeze()
        x = x.astype(mindspore.int32)

        out = self.embedding(x)
        # out = norm(out)

        return out

class FactorizationMachine(nn.Cell):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def construct(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = ops.ReduceSum()(x, 1) ** 2
        sum_of_square = ops.ReduceSum()(x ** 2, 1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = ops.ReduceSum(keep_dims=True)(ix, 1)
        return 0.5 * ix







class FactorizationMachineModel(nn.Cell):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        # self.field_dims = field_dims
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fm = FactorizationMachine(reduce_sum=True)


    def construct(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """

        out = self.linear(x) + self.fm(self.embedding(x))

        out =  1/(1 +ops.Exp()(-out))
        out = ops.Squeeze(1)(out)


        return out

