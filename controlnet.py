class ZeroConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel=1)
        ## здесь нужно проинициализировать свертку нулями 

    def forward(self, x):
        return self.conv(x)

class ControlCUNet(nn.Module):
    def __init__(self, cunet):
        super().__init__()
        self.cunet = cunet
        self.in_channels = cunet.in_channels
        self.out_channels = cunet.out_channels
        self.noise_channels = cunet.noise_channels
        self.base_factor = cunet.base_factor

        factor = 2
        self.zero0 = ...
        self.inc = ...
        self.zero_inc = ...
        self.down1 = = ...
        self.zero_down1 = ...
        self.down2 = ...
        self.zero_down2 = ...
        self.down3 = ...
        self.zero_down3 = ...
        self.down4 = ...
        self.zero_down4 = ...

        # важно оставить такими же названия повторяющихся модулей, чтобы копирование сработало
        misc.copy_params_and_buffers(src_module=self.cunet, dst_module=self, require_all=False)
        for param in self.cunet.parameters():
            param.requires_grad = False

    def forward(self, x, noise_labels, class_labels, cond=None):
        if cond is None:
            return self.cunet(x, noise_labels, class_labels)

        emb = self.cunet.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.cunet.map_label is not None:
            tmp = class_labels
            emb = emb + self.cunet.map_label(tmp * np.sqrt(self.cunet.map_label.in_features))

        emb = silu(self.cunet.map_layer0(emb))
        z = silu(self.cunet.map_layer1(emb)).unsqueeze(-1).unsqueeze(-1)

        x1 = self.cunet.inc(x)
        x2 = self.cunet.down1(x1)
        x3 = self.cunet.down2(x2)
        x4 = self.cunet.down3(x3)
        x5 = self.cunet.down4(x4)

        # your code here
        out = ...
        return out
