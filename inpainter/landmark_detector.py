

class LandmarkDetector():
    def __init__(self, model, to_fan, device='cuda'):
        self.model = model
        self.to_fan = to_fan
        self.size = 256
        self.device=device


    def __call__(self, fake_image):
        fake_image = self.to_fan(fake_image)
        center = torch.FloatTensor([self.size/2, self.size/2]).to(self.device)
        center = center.unsqueeze(0)
        scale = torch.FloatTensor([1]).to(self.device).unsqueeze(0)

        out, _ = self.model(fake_image)
        pts = self.heatmaps_to_coords(out)

        # pts_img = self.scale_preds(pts, center, scale)
        # pts, pts_img = pts * 4, pts_img

        # detected_landmarks = pts_img
        return pts


    def heatmaps_to_coords(self, heatmaps, normalize=False):
        xy = self.dsnt(heatmaps)
        x, y = xy.split(1, -1)
        coords = torch.cat([x, y], -1)

        # Denormalize
        if not normalize:
            coords = (coords + 1) / 2
            dim = heatmaps.shape[-2:]
            for n, coord in zip(dim, coords.split(1, -1)):
                coord *= (n - 1)

        return coords

    def _normalized_linspace(self, length, dtype=None, device=None):
        first = -(length - 1) / length
        last = (length - 1) / length
        return torch.linspace(first, last, length, dtype=dtype, device=device)

    def _coord_expectation(self, heatmaps, dim, transform=None):
        dim_size = heatmaps.size()[dim]
        own_coords = self._normalized_linspace(dim_size, dtype=heatmaps.dtype, device=heatmaps.device)
        if transform:
            own_coords = transform(own_coords)
        summed = heatmaps.view(-1, *heatmaps.size()[2:])
        for i in range(2 - heatmaps.dim(), 0):
            if i != dim:
                summed = summed.sum(i, keepdim=True)
        summed = summed.view(summed.size(0), -1)
        expectations = summed.mul(own_coords.view(-1, own_coords.size(-1))).sum(-1)
        expectations = expectations.view(*heatmaps.size()[:2])
        return expectations

    def dsnt(self, heatmaps):
        dim_range = range(-1, 1 - heatmaps.dim(), -1)
        mu = torch.cat([self._coord_expectation(heatmaps, dim).unsqueeze(-1) for dim in dim_range], -1)
        return mu

    def scale_preds(self, preds, center=None, scale=None, res=64):
        preds_scaled = torch.zeros(preds.size())
        if center is not None and scale is not None:
            for b in range(center.size(0)):
                transMat = self.getTransform(center[b], scale[b], res)
                for i in range(preds[b].size(0)):
                    preds_scaled[b, i] = self.transform(preds[b, i], transMat, True)
        return preds_scaled

    def getTransform(self, center, scale, resolution, rotate=0):
        h = 200.0 * scale
        t = torch.eye(4)

        # scale
        t[0, 0] = resolution / h
        t[1, 1] = resolution / h
        t[2, 2] = resolution / h

        # transform
        t[0, 3] = resolution * (-center[0] / h + 0.5)
        t[1, 3] = resolution * (-center[1] / h + 0.5)

        # rotation
        if not rotate == 0:
            rotate = -rotate  # To match direction of rotation from cropping
            rot_mat = torch.eye(4)
            rot_rad = rotate * np.pi / 180
            sn, cs = torch.sin(rot_rad), torch.cos(rot_rad)
            rot_mat[:2, :2] = torch.tensor([[cs, -sn],
                                            [sn, cs]])

            # Need to rotate around center
            t_mat = torch.eye(4)
            t_mat[:2, 3] = -resolution / 2
            t_inv = t_mat.clone()
            t_inv[:2, 3] *= -1
            t = torch.matmul(t_inv, torch.matmul(rot_mat, torch.matmul(t_mat, t)))

        return t

    def transform(self, point, transform, invert=False):
        dim = len(point)
        _pt = torch.ones(4)
        for idx in range(0,dim):
            _pt[idx] = point[idx]

        if invert:
            transform = torch.inverse(transform)

        new_point = (torch.matmul(transform, _pt))

        #Hnormalize
        new_point = torch.div(new_point[0:dim], new_point[-1])

        new_point[0:2] = new_point[0:2]#.int() + 1

        return new_point



################################ FAN
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features))

        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x


class FAN(nn.Module):

    def __init__(self, num_modules=1, super_res=False):
        super(FAN, self).__init__()
        self.num_modules = num_modules
        self.super_res = super_res

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                            68, kernel_size=1, stride=1, padding=0))
            # self.add_module('dropout_' + str(hg_module), nn.Dropout2d(p=0.1))

            # if hg_module < self.num_modules - 1:
            self.add_module(
                'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('al' + str(hg_module), nn.Conv2d(68,
                                                             256, kernel_size=1, stride=1, padding=0))

        if self.super_res:
            input_skip_seq = [
                conv3x3(3, 128),
                nn.BatchNorm2d(128),
            ]
            self.input_skip = nn.Sequential(*input_skip_seq)

            downsample_seq = [
                conv3x3(256, 256),
                nn.BatchNorm2d(256),
            ]
            self.downsample_layer = nn.Sequential(*downsample_seq)

            # output
            up_sequence = [
                ConvBlock(256, 256),
                Interpolate(size=(128,128), mode='bilinear'), # 64 -> 128
                conv3x3(256, 128),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                Interpolate(size=(256, 256), mode='bilinear'), # 128 -> 256
            ]
            self.up_layer = nn.Sequential(*up_sequence)

            output_sequence = [
                conv3x3(128, 128),
                nn.BatchNorm2d(128),
                conv3x3(128, 128),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 68, kernel_size=1, stride=1, padding=0)
            ]
            self.output_layer = nn.Sequential(*output_sequence)

    def forward(self, input):
        x = F.relu(self.bn1(self.conv1(input)), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            # ll = self._modules['dropout_' + str(i)](ll)

            # Predict heatmaps
            # tmp_out = self._modules['l' + str(i)](ll)
            tmp_out = torch.sigmoid(self._modules['l' + str(i)](ll))
            outputs.append(tmp_out)

            ll = self._modules['bl' + str(i)](ll)
            tmp_out_ = self._modules['al' + str(i)](tmp_out)

            if i < self.num_modules - 1:
                previous = previous + ll + tmp_out_

        if self.super_res:
            # x = self.downsample_layer(x)
            out = self.downsample_layer(x) + ll + tmp_out_
            out = self.up_layer(out)
            out = self.input_skip(input) + out
            out = self.output_layer(out)

            # return out, outputs
            return torch.sigmoid(out), outputs

        return outputs[-1], outputs

