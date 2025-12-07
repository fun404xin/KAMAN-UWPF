import torch
import torch.nn.functional as F
import math

class KANLinear_Haar(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,  # 使用小波层数
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.ReLU
    ):
        super(KANLinear_Haar, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.num_wavelet_basis = grid_size * 2  # 每层两个基函数（mask1, mask2）

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features, self.num_wavelet_basis))
        self.spline_scaler = torch.nn.Parameter(torch.ones(out_features, in_features))

        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        torch.nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5) * self.scale_spline)

    def haar_wavelet_bases(self, x: torch.Tensor):
        # 归一化到 [0,1]
        x_min = x.min(dim=0, keepdim=True)[0]
        x_max = x.max(dim=0, keepdim=True)[0]
        x = (x - x_min) / (x_max - x_min + 1e-8)  # [batch, in_features]

        batch_size = x.shape[0]
        bases = []

        for level in range(self.grid_size):
            scale = 2 ** level
            for shift in range(scale):
                left = shift / scale
                mid = (shift + 0.5) / scale
                right = (shift + 1.0) / scale
                mask1 = ((x >= left) & (x < mid)).float()
                mask2 = ((x >= mid) & (x < right)).float()
                base = mask1 - mask2
                bases.append(base)  # [batch, in_features]

        bases = torch.stack(bases, dim=-1)  # [batch, in_features, num_basis]
        return bases

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)  # [batch, out]
        wavelet_output = F.linear(
            self.haar_wavelet_bases(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + wavelet_output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / (regularization_loss_activation + 1e-8)
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class HybridKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.ReLU,
    ):
        super(HybridKAN, self).__init__()
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear_Haar(
                    in_features=in_features,
                    out_features=out_features,
                    grid_size=grid_size,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


