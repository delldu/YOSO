import math
import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple
from torchvision.ops import deform_conv2d
import pdb

class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    def __new__(cls, *, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)


# class DeformConv(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         stride=1,
#         padding=0,
#         dilation=1,
#         groups=1,
#         deformable_groups=1,
#         bias=False,
#         norm=None,
#         activation=None,
#     ):
#         """
#         Deformable convolution from :paper:`deformconv`.

#         Arguments are similar to :class:`Conv2D`. Extra arguments:

#         Args:
#             deformable_groups (int): number of groups used in deformable convolution.
#             norm (nn.Module, optional): a normalization layer
#             activation (callable(Tensor) -> Tensor): a callable activation function
#         """
#         super(DeformConv, self).__init__()

#         assert not bias
#         assert in_channels % groups == 0, "in_channels {} cannot be divisible by groups {}".format(
#             in_channels, groups
#         )
#         assert (
#             out_channels % groups == 0
#         ), "out_channels {} cannot be divisible by groups {}".format(out_channels, groups)

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = (kernel_size, kernel_size)
#         self.stride = (stride, stride)
#         self.padding = (padding, padding)
#         self.dilation = (dilation, dilation)
#         self.groups = groups
#         self.deformable_groups = deformable_groups
#         self.norm = norm
#         self.activation = activation

#         self.weight = nn.Parameter(
#             torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size)
#         )
#         self.bias = None

#         nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

#     def forward(self, x, offset):
#         if x.numel() == 0:
#             # When input is empty, we want to return a empty tensor with "correct" shape,
#             # So that the following operations will not panic
#             # if they check for the shape of the tensor.
#             # This computes the height and width of the output tensor
#             output_shape = [
#                 (i + 2 * p - (di * (k - 1) + 1)) // s + 1
#                 for i, p, di, k, s in zip(
#                     x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
#                 )
#             ]
#             output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
#             return _NewEmptyTensorOp.apply(x, output_shape)

#         x = deform_conv(
#             x,
#             offset,
#             self.weight,
#             self.stride,
#             self.padding,
#             self.dilation,
#             self.groups,
#             self.deformable_groups,
#         )
#         if self.norm is not None:
#             x = self.norm(x)
#         if self.activation is not None:
#             x = self.activation(x)
#         return x

#     def extra_repr(self):
#         tmpstr = "in_channels=" + str(self.in_channels)
#         tmpstr += ", out_channels=" + str(self.out_channels)
#         tmpstr += ", kernel_size=" + str(self.kernel_size)
#         tmpstr += ", stride=" + str(self.stride)
#         tmpstr += ", padding=" + str(self.padding)
#         tmpstr += ", dilation=" + str(self.dilation)
#         tmpstr += ", groups=" + str(self.groups)
#         tmpstr += ", deformable_groups=" + str(self.deformable_groups)
#         tmpstr += ", bias=False"
#         return tmpstr


class ModulatedDeformConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=True,
        norm=None,
        activation=None,
    ):
        """
        Modulated deformable convolution from :paper:`deformconv2`.

        Arguments are similar to :class:`Conv2D`. Extra arguments:

        Args:
            deformable_groups (int): number of groups used in deformable convolution.
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        """
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.norm = norm
        self.activation = activation

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x, offset, mask):
        if x.numel() == 0:
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            return _NewEmptyTensorOp.apply(x, output_shape)

        # x = modulated_deform_conv(
        #     x,
        #     offset,
        #     mask,
        #     self.weight,
        #     self.bias,
        #     self.stride,
        #     self.padding,
        #     self.dilation,
        #     self.groups,
        #     self.deformable_groups,
        # )
        x = deform_conv2d(
            x,
            offset, 
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            mask,
            # self.groups,
            # self.deformable_groups,
        )

        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        tmpstr = "in_channels=" + str(self.in_channels)
        tmpstr += ", out_channels=" + str(self.out_channels)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", padding=" + str(self.padding)
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", groups=" + str(self.groups)
        tmpstr += ", deformable_groups=" + str(self.deformable_groups)
        tmpstr += ", bias=" + str(self.with_bias)
        return tmpstr


class DeformLayer(nn.Module):
    def __init__(self, in_planes, out_planes, deconv_kernel=4, deconv_stride=2, deconv_pad=1, deconv_out_pad=0, 
        modulate_deform=True, num_groups=1, deform_num_groups=1, dilation=1):
        super(DeformLayer, self).__init__()
        self.deform_modulated = modulate_deform

        deform_conv_op = ModulatedDeformConv
        # offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size
        offset_channels = 27
        # else:
        #     deform_conv_op = DeformConv
        #     offset_channels = 18
        
        self.dcn_offset = nn.Conv2d(in_planes,
                                    offset_channels * deform_num_groups,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1*dilation,
                                    dilation=dilation)
        self.dcn = deform_conv_op(in_planes,
                                  out_planes,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1*dilation,
                                  bias=False,
                                  groups=num_groups,
                                  dilation=dilation,
                                  deformable_groups=deform_num_groups)
        
        self.dcn_bn = nn.SyncBatchNorm(out_planes) # nn.GroupNorm(64, out_planes) # nn.BatchNorm2d(out_planes) #
        self.up_sample = nn.ConvTranspose2d(in_channels=out_planes,
                                            out_channels=out_planes,
                                            kernel_size=deconv_kernel,
                                            stride=deconv_stride, padding=deconv_pad,
                                            output_padding=deconv_out_pad,
                                            bias=False)
        self._deconv_init()
        self.up_bn = nn.SyncBatchNorm(out_planes) # nn.GroupNorm(64, out_planes) # nn.BatchNorm2d(out_planes) #
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        if self.deform_modulated:
            offset_mask = self.dcn_offset(out)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            out = self.dcn(out, offset, mask)
        else:
            offset = self.dcn_offset(out)
            out = self.dcn(out, offset)
        x = out
        
        x = self.dcn_bn(x)
        x = self.relu(x)
        x = self.up_sample(x)
        x = self.up_bn(x)
        x = self.relu(x)
        return x

    def _deconv_init(self):
        w = self.up_sample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]


class LiteDeformConv(nn.Module):
    def __init__(self, backbone_shape):
        super(LiteDeformConv, self).__init__()

        in_features = ['res2', 'res3', 'res4', 'res5']
        in_channels = []
        out_channels = [128]
        for feat in in_features:
            tmp = backbone_shape[feat].channels
            in_channels.append(tmp)
            out_channels.append(tmp//2)
        
        self.lateral_conv0 = nn.Conv2d(in_channels=in_channels[-1], out_channels=out_channels[-1], 
            kernel_size=1, stride=1, padding=0)
        self.deform_conv1 = DeformLayer(in_planes=out_channels[-1], out_planes=out_channels[-2])
        self.lateral_conv1 = nn.Conv2d(in_channels=in_channels[-2], out_channels=out_channels[-2], 
            kernel_size=1, stride=1, padding=0)

        self.deform_conv2 = DeformLayer(in_planes=out_channels[-2], out_planes=out_channels[-3])
        self.lateral_conv2 = nn.Conv2d(in_channels=in_channels[-3], out_channels=out_channels[-3], 
            kernel_size=1, stride=1, padding=0)
        
        self.deform_conv3 = DeformLayer(in_planes=out_channels[-3], out_planes=out_channels[-4])
        self.lateral_conv3 = nn.Conv2d(in_channels=in_channels[-4], out_channels=out_channels[-4], 
            kernel_size=1, stride=1, padding=0)
        
        self.output_conv = nn.Conv2d(in_channels=out_channels[-5], out_channels=out_channels[-5], 
            kernel_size=3, stride=1, padding=1)
        
        self.bias = nn.Parameter(torch.FloatTensor(1,out_channels[-5],1,1), requires_grad=True)
        self.bias.data.fill_(0.0)
        
        self.conv_a5 = nn.Conv2d(in_channels=out_channels[-1], out_channels=out_channels[-5], 
            kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_a4 = nn.Conv2d(in_channels=out_channels[-2], out_channels=out_channels[-5], 
            kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_a3 = nn.Conv2d(in_channels=out_channels[-3], out_channels=out_channels[-5], 
            kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_a2 = nn.Conv2d(in_channels=out_channels[-4], out_channels=out_channels[-5], 
            kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, features_list):
        p5 = self.lateral_conv0(features_list[-1])
        x5 = p5
        x = self.deform_conv1(x5)

        p4 = self.lateral_conv1(features_list[-2])
        x4 = p4 + x
        x = self.deform_conv2(x4)

        p3 = self.lateral_conv2(features_list[-3])
        x3 = p3 + x
        x = self.deform_conv3(x3)

        p2 = self.lateral_conv3(features_list[-4])
        x2 = p2 + x
        
        # CFA
        x5 = F.interpolate(self.conv_a5(x5), scale_factor=8, align_corners=False, mode='bilinear')
        x4 = F.interpolate(self.conv_a4(x4), scale_factor=4, align_corners=False, mode='bilinear')
        x3 = F.interpolate(self.conv_a3(x3), scale_factor=2, align_corners=False, mode='bilinear')
        x2 = self.conv_a2(x2)
        x = x5 + x4 + x3 + x2 + self.bias
        
        x = self.output_conv(x)

        # IFA
        # x5 = F.interpolate(x5, scale_factor=8, align_corners=False, mode='bilinear')
        # x4 = F.interpolate(x4, scale_factor=4, align_corners=False, mode='bilinear')
        # x3 = F.interpolate(x3, scale_factor=2, align_corners=False, mode='bilinear')
        # x_fuse = torch.concat([x5,x4,x3,x2], dim=1)
        # x_fuse = self.fuse_conv(x_fuse)

        # x = self.output_conv(x_fuse)

        return x


class YOSONeck(nn.Module):
    def __init__(self):
        super().__init__()
        backbone_shape = {
            'res2': ShapeSpec(channels=256, height=None, width=None, stride=4), 
            'res3': ShapeSpec(channels=512, height=None, width=None, stride=8), 
            'res4': ShapeSpec(channels=1024, height=None, width=None, stride=16), 
            'res5': ShapeSpec(channels=2048, height=None, width=None, stride=32)
        }

        self.deconv = LiteDeformConv(backbone_shape=backbone_shape)
        self.loc_conv = nn.Conv2d(in_channels=128+2, #backbone_shape['res2'].channels//2 + 2,
                                  out_channels=256, #cfg.MODEL.YOSO.HIDDEN_DIM,
                                  kernel_size=1,
                                  stride=1)
        self._reset_parameters()
        # pdb.set_trace()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_coord(self, input_feat):
        x_range = torch.linspace(-1, 1, input_feat.shape[-1], device=input_feat.device)
        y_range = torch.linspace(-1, 1, input_feat.shape[-2], device=input_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([input_feat.shape[0], 1, -1, -1])
        x = x.expand([input_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        return coord_feat

    def forward(self, features_list):
        features = self.deconv(features_list)
        coord_feat = self.generate_coord(features)
        features = torch.cat([features, coord_feat], 1)
        features = self.loc_conv(features)
        return features