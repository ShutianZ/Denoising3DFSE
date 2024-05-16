import torch
import torch.nn as nn
import torch.nn.functional as F

class Denoise2D(nn.Module):
    def __init__(self,in_channels=4,out_channel=2, n_channels=64,depth=4,kernel_size=3, stride=1, padding=1):
        super(Denoise2D, self).__init__()
        self.common1 = DoubleModule(in_channels,n_channels*2,depth,kernel_size,stride,padding)
        self.M14_1 = nn.Conv2d(in_channels = n_channels*2, out_channels = out_channel, kernel_size = kernel_size, padding = padding,bias=True)
        self.M14_2 = nn.Sequential(
            nn.Conv2d(in_channels = out_channel, out_channels = n_channels, kernel_size = kernel_size, padding = padding, bias=True),
            nn.ReLU(inplace = True)
        )
        self.M19 = nn.Sequential(
            nn.Conv2d(in_channels = n_channels*2, out_channels = n_channels, kernel_size = kernel_size, padding = padding, bias=True),
            nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95),
            nn.ReLU(inplace = True)
        )
        self.common2 = DoubleM(n_channels*2,out_channel,depth,kernel_size,stride,padding)
        
    def forward(self, x,y):
        identity = (x+y)/2
        in_v = torch.cat((x,y),1)
        out_1 = self.common1(in_v)
        
        M14_2out = self.M14_1(out_1) + identity
        M14_3out = self.M14_2(M14_2out)
        
        M19_2out = self.M19(out_1)
        
        out = self.common2(torch.cat((M14_3out,M19_2out),1)) + M14_2out
        return out
    
class DoubleModule(nn.Module):
    def __init__(self,in_channels,n_channels,depth,kernel_size, stride, padding):
        super().__init__()
        self.stride = stride
        layers = []
        layers.append(nn.Conv2d(in_channels = in_channels, out_channels = n_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = True,dilation=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth):
            layers.append(nn.Conv2d(in_channels = n_channels, out_channels = n_channels, kernel_size = kernel_size, padding = padding, bias=True,dilation=1))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace = True))
            
        layers.append(nn.Conv2d(in_channels = n_channels, out_channels = n_channels, kernel_size = kernel_size, padding = padding, bias=True,dilation=1))
        layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
        layers.append(nn.ReLU(inplace = True))
        
        self.dr = nn.Sequential(*layers)
        
        self._initialize_weights()

    def forward(self, x):
        return self.dr(x) 
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
                
class DoubleM(nn.Module):
    def __init__(self,n_channels,out_channel,depth,kernel_size, stride, padding):
        super().__init__()
        self.stride = stride
        layers = []
        for _ in range(depth):
            layers.append(nn.Conv2d(in_channels = n_channels, out_channels = n_channels, kernel_size = kernel_size, padding = padding, bias=True,dilation=1))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace = True))
            
        layers.append(nn.Conv2d(in_channels = n_channels, out_channels = n_channels, kernel_size = kernel_size, padding = padding, bias=True,dilation=1))
        layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
        layers.append(nn.ReLU(inplace = True))
        
        layers.append(nn.Conv2d(in_channels = n_channels, out_channels = out_channel, kernel_size = kernel_size, padding = padding,bias=True,dilation=1))
        
        self.dr = nn.Sequential(*layers)
        
        self._initialize_weights()

    def forward(self, x):
        return self.dr(x) 
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
                