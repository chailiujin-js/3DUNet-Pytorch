import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(UNet, self).__init__()
        self.training = training
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4 = nn.Conv3d(128, 256, 3, stride=1, padding=1)

        self.decoder2 = nn.Conv3d(256, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv3d(128, 64, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(32, out_channel, 3, stride=1, padding=1)

        self.map4 = nn.Sequential(
            nn.Conv3d(2, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Softmax(dim=1)
        )

        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Softmax(dim=1)
        )

        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Softmax(dim=1)
        )

        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(16, 32, 32), mode='trilinear'),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = F.relu(F.max_pool3d(self.encoder1(x), 2, 2))
        t1 = out
        out = F.relu(F.max_pool3d(self.encoder2(out), 2, 2))
        t2 = out
        out = F.relu(F.max_pool3d(self.encoder3(out), 2, 2))
        t3 = out
        out = F.relu(F.max_pool3d(self.encoder4(out), 2, 2))

        output1 = self.map1(out)
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2, 2), mode='trilinear'))
        out += t3
        output2 = self.map2(out)
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2, 2), mode='trilinear'))
        out += t2
        output3 = self.map3(out)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2, 2), mode='trilinear'))
        out += t1

        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2, 2), mode='trilinear'))
        output4 = self.map4(out)

        return (output1, output2, output3, output4) if self.training else output4


if __name__ == "__main__":
    import time
    import torch

    # 指定使用第一个 GPU
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.Tensor(1, 1, 48, 256, 256).to(device)  # 修正这里
    model = UNet(in_channel=1).to(device)  # 将模型移动到设备
    out = model(x)
    print("out size: ", [o.shape for o in out])  # 打印每个输出的形状

    end_time = time.time()
    cost_time = end_time - start_time
    print(f"代码执行时间：{cost_time:.2f}秒")
