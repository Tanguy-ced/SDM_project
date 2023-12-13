from torch import nn
import torch
import torch.nn.functional as F

class Inception(nn.Module):
    
    def __init__(self, n_labels = 100, n_input = 4, dropout = 0.5, last_layer = True, logit=False, sigmoid=False,  exp=False, normalize_weight=1., temperature=1., kernel = 11):
        
        super(Inception, self).__init__()
        if n_input >= 15:
            self.Conv2d_1a_3x3 = BasicConv2d(n_input, 80, kernel_size=3, stride=1, padding=1)
            self.Conv2d_2a_3x3 = BasicConv2d(80, 80, kernel_size=3, stride=1, padding=1)
            self.Conv2d_2b_3x3 = BasicConv2d(80, 100, kernel_size=3, padding=1, stride=1)
            self.Conv2d_3b_1x1 = BasicConv2d(100, 124, kernel_size=1)
            self.Conv2d_4a_3x3 = BasicConv2d(124, 192, kernel_size=3, padding=1, stride=1)
        else:
            self.Conv2d_1a_3x3 = BasicConv2d(n_input, 32, kernel_size=3, stride=1, padding=1)
            self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
            self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1, stride=1)
            self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
            self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3, padding=1, stride=1)
            
        
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, n_labels)

        self.dropout = dropout
        self.last_layer = last_layer
        self.logit = logit
        self.sigmoid = sigmoid
        self.exp = exp
        self.kernel = kernel

        self.temperature = temperature

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # if poisson loss, for instance...
        if normalize_weight != 1.:
            for p in self.parameters():
                p.data.div_(normalize_weight)

    def forward(self, x ,val = True):
            # (80, 32) x 256 x 256
            #if val: print(x.shape)
            x = self.Conv2d_1a_3x3(x)
            #if val: print(x.shape)
            # (124, 32) x 256 x 256
            x = self.Conv2d_2a_3x3(x)
            #if val: print(x.shape)
            # (124, 32) x 256 x 256
            x = self.Conv2d_2b_3x3(x)
            #if val: print(x.shape)
            # (124, 32) x 256 x 256
            x = self.Conv2d_3b_1x1(x)
            #if val: print(x.shape)
            # (124, 80) x 256 x 256
            x = self.Conv2d_4a_3x3(x)
            #if val: print(x.shape)
            # 192 x 256 x 256
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            #if val: print(x.shape)
            # 192 x 128 x 128
            x = self.Mixed_5b(x)
            #if val: print(x.shape)
            # 256 x 128 x 128
            x = self.Mixed_5c(x)

            # 288 x 128 x 128
            x = self.Mixed_5d(x)

            # 288 x 128 x 128
            x = self.Mixed_6a(x)

            # 768 x 63 x 63
            x = self.Mixed_6b(x)

            # 768 x 63 x 63
            x = self.Mixed_6c(x)

            # 768 x 63 x 63
            x = self.Mixed_6d(x)

            # 768 x 63 x 63
            x = self.Mixed_6e(x)

            # 768 x 63 x 63
            x = self.Mixed_7a(x)

            # 1280 x 31 x 31
            x = self.Mixed_7b(x)

            # 2048 x 31 x 31
            x = self.Mixed_7c(x)
            if val: print("Hey you",x.shape,x.size(2))
            # 2048 x 31 x 31
            x = F.avg_pool2d(x, kernel_size=x.size(2))

            # 1 x 1 x 2048
            x = F.dropout(x, p=self.dropout, training=self.training)  # increased dropout probability

            # 1 x 1 x 2048
            x = x.view(x.size(0), -1)

            if self.last_layer:
                # 2048
                x = self.fc(x)
                # (num_classes)
                if not self.training and not self.logit and not self.exp and not self.sigmoid:
                    x = F.softmax(x*self.temperature, dim=-1)
                elif not self.training and not self.logit and self.exp:
                    x = x.exp()
                elif not self.training and not self.logit and self.sigmoid:
                    x = F.sigmoid(x)

            return x
        
    def __repr__(self):
        return '(Environmental Inception)'


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
    
class Two_branch_Inception(nn.Module):
    
    def __init__(self, n_labels = 100, n_input = 4, dropout = 0.5, last_layer = True, logit=False, sigmoid=False,  exp=False, normalize_weight=1., temperature=1.):
        
        
        super(Two_branch_Inception, self).__init__()
        
        self.model_rgb = Inception(n_input=4 , n_labels= n_labels ,kernel =11)
        self.model_env = Inception(n_input = n_input, n_labels= n_labels , kernel = 4 )
        
        

    def forward(self, rgb_x, env_x, val=False):
            
            
            if val: print("Processing RGB images" , rgb_x.shape)
            rgb_x = self.model_rgb(rgb_x.to(torch.float32))
            if val: print("Processing Env Cov" , rgb_x.shape)
            env_x = self.model_env(env_x.to(torch.float32))
            
            print(rgb_x.shape , env_x.shape)
            
            return rgb_x + env_x
            
class Inception_Env(nn.Module):
    
    def __init__(self, n_labels = 100, n_input = 4, dropout = 0.5, last_layer = True, logit=False, sigmoid=False,  exp=False, normalize_weight=1., temperature=1.):
        
        
        super(Two_branch_Inception, self).__init__()
        
        self.model_env = Inception(n_input = n_input, n_labels= n_labels , kernel = 4 )
        
        

    def forward(self, env_x, val=False):
            
        
        if val: print("Processing Env Cov" , env_x.shape)
        env_x = self.model_env(env_x.to(torch.float32))
        
        print(env_x.shape)
        
        return env_x
            
class Inception_RGB(nn.Module):
    
    def __init__(self, n_labels = 100, n_input = 4, dropout = 0.5, last_layer = True, logit=False, sigmoid=False,  exp=False, normalize_weight=1., temperature=1.):
        
        
        super(Inception_RGB, self).__init__()
        self.model_rgb = Inception(n_input=4 , n_labels= n_labels ,kernel =11)

    def forward(self, rgb_x, val=False):
            
            
            if val: print("Processing RGB images" , rgb_x.shape)
            rgb_x = self.model_rgb(rgb_x.to(torch.float32))
            return rgb_x


        
    
# class cnn(nn.Module):
#     def __init__(self, n_features, n_species):
#         super().__init__()
#         self.conv1 = nn.Conv2d(n_features, 32, kernel_size=3)
#         self.act1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=3)
#         self.drop1 = nn.Dropout(0.3)

#         self.conv2 = nn.Conv2d(32, 8, kernel_size=3)
#         self.act2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2)

#         self.flat = nn.Flatten()

#         self.fc4 = nn.Linear(3200, n_species)
    
#     def forward(self, x):
#         # input n_featuresx128x128, output 32x126x126
#         x = self.act1(self.conv1(x))
#         # input 32x126x126, output 32x42x42
#         x = self.pool1(x)
#         x = self.drop1(x)

#         # input 32x42x42, output 8x40x40
#         x = self.act2(self.conv2(x))
#         # input 8x40x40, output 8x20x20
#         x = self.pool2(x)
#         # input 8x20x20, output 3200
#         x = self.flat(x)
        
#         # input 3200, output n_species
#         x = self.fc4(x)
#         return x

# class cnn_batchnorm(nn.Module):
#     def __init__(self, n_features, n_species):
#         super().__init__()
#         self.conv1 = nn.Conv2d(n_features, 32, kernel_size=3)
#         self.batchnorm1 = nn.BatchNorm2d(32)
#         self.act1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=3)
#         self.drop1 = nn.Dropout(0.3)

#         self.conv2 = nn.Conv2d(32, 8, kernel_size=3)
#         self.batchnorm2 = nn.BatchNorm2d(8)
#         self.act2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2)

#         self.flat = nn.Flatten()

#         self.fc3 = nn.Linear(3200, 512)
#         self.batchnorm3 = nn.BatchNorm1d(512)
#         self.act3 = nn.ReLU()
#         self.drop3 = nn.Dropout(0.5)

#         self.fc4 = nn.Linear(512, n_species)
    
#     def forward(self, x):
#         # input n_featuresx128x128, output 32x126x126
#         x = self.act1(self.batchnorm1(self.conv1(x)))
#         # input 32x126x126, output 32x42x42
#         x = self.pool1(x)
#         x = self.drop1(x)

#         # input 32x42x42, output 8x40x40
#         x = self.act2(self.batchnorm2(self.conv2(x)))
#         # input 8x40x40, output 8x20x20
#         x = self.pool2(x)
#         # input 8x20x20, output 3200
#         x = self.flat(x)

#         # input 3200, output 512
#         x = self.act3(self.batchnorm3(self.fc3(x)))
#         x = self.drop3(x)
        
#         # input 512, output n_species
#         x = self.fc4(x)
#         return x

# class cnn_batchnorm_act(nn.Module):
#     def __init__(self, n_features, n_species):
#         super().__init__()
#         self.conv1 = nn.Conv2d(n_features, 32, kernel_size=3)
#         self.batchnorm1 = nn.BatchNorm2d(32)
#         self.act1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=3)
#         self.drop1 = nn.Dropout(0.3)

#         self.conv2 = nn.Conv2d(32, 8, kernel_size=3)
#         self.batchnorm2 = nn.BatchNorm2d(8)
#         self.act2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2)

#         self.flat = nn.Flatten()

#         self.fc3 = nn.Linear(3200, 512)
#         self.batchnorm3 = nn.BatchNorm1d(512)
#         self.act3 = nn.ReLU()
#         self.drop3 = nn.Dropout(0.5)

#         self.fc4 = nn.Linear(512, n_species)
#         self.act4 = nn.ReLU()
    
#     def forward(self, x):
#         # input n_featuresx128x128, output 32x126x126
#         x = self.act1(self.batchnorm1(self.conv1(x)))
#         # input 32x126x126, output 32x42x42
#         x = self.pool1(x)
#         x = self.drop1(x)

#         # input 32x42x42, output 8x40x40
#         x = self.act2(self.batchnorm2(self.conv2(x)))
#         # input 8x40x40, output 8x20x20
#         x = self.pool2(x)
#         # input 8x20x20, output 3200
#         x = self.flat(x)

#         # input 3200, output 512
#         x = self.act3(self.batchnorm3(self.fc3(x)))
#         x = self.drop3(x)
        
#         # input 512, output n_species
#         x = self.act4(self.fc4(x))
#         return x

# class cnn_batchnorm_patchsize_20(nn.Module):
#     def __init__(self, n_features, n_species, dropout=0.3):
#         super().__init__()
#         self.conv1 = nn.Conv2d(n_features, 16, kernel_size=3)
#         self.batchnorm1 = nn.BatchNorm2d(16)
#         self.act1 = nn.ReLU()
#         self.drop1 = nn.Dropout(dropout)

#         self.conv2 = nn.Conv2d(16, 8, kernel_size=3)
#         self.batchnorm2 = nn.BatchNorm2d(8)
#         self.act2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2)
#         self.drop2 = nn.Dropout(dropout)

#         self.flat = nn.Flatten()

#         self.fc3 = nn.Linear(512, 1024)
#         self.batchnorm3 = nn.BatchNorm1d(1024)
#         self.act3 = nn.ReLU()
#         self.drop3 = nn.Dropout(dropout)

#         self.fc4 = nn.Linear(1024, n_species)
    
#     def forward(self, x):
#         # input n_featuresx20x20, output 16x18x18
#         x = self.act1(self.batchnorm1(self.conv1(x)))
#         x = self.drop1(x)

#         # input 16x18x18, output 8x16x16
#         x = self.act2(self.batchnorm2(self.conv2(x)))
#         # input 8x16x16, output 8x8x8
#         x = self.pool2(x)
#         x = self.drop2(x)
#         # input 8x8x8, output 512
#         x = self.flat(x)

#         # input 512, output 1024
#         x = self.act3(self.batchnorm3(self.fc3(x)))
#         x = self.drop3(x)
        
#         # input 1024, output n_species
#         x = self.fc4(x)
#         return x
    
# class MLP(nn.Module):

#     def __init__(self, input_size, output_size, num_layers, width, dropout=0.0):
#         super(MLP, self).__init__()

#         self.output_size = output_size
#         self.width = width

#         layers = []

#         layers.append(nn.Linear(input_size, width))
#         layers.append(nn.SiLU())

#         for _ in range(num_layers - 1):
#             layers.append(nn.BatchNorm1d(width))
#             layers.append(nn.Linear(width, width))
#             layers.append(nn.SiLU())
#             layers.append(nn.Dropout(p=dropout))
    
#         layers.append(nn.Linear(width, output_size))

#         self.layers = nn.Sequential(*layers)

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x

class twoBranchCNN(nn.Module):

    def __init__(self, n_species):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=5)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(8, 8, kernel_size=5)
        self.batchnorm2 = nn.BatchNorm2d(8)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=4)
        self.flat2 = nn.Flatten()

        self.conv3 = nn.Conv2d(21, 16, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(16, 16, kernel_size=3)
        self.batchnorm4 = nn.BatchNorm2d(16)
        self.act4 = nn.ReLU()
        self.flat4 = nn.Flatten()

        self.fc5 = nn.Linear(1544, 1024)
        self.act5 = nn.ReLU()

        self.fc6 = nn.Linear(1024, n_species)

    def forward(self, rgb_x, env_x, val=True):
        if val: print("entering top network",rgb_x.shape)
        # input 4x100x100 -> output 4x96x96 (k=5)
        rgb_x = self.act1(self.batchnorm1(self.conv1(rgb_x)))
        # input 8x96x96 -> output 8x48x48 (k=2)
        rgb_x = self.pool1(rgb_x)
        # input 8x48x48 -> output 8x44x44 (k=5)
        rgb_x = self.act2(self.batchnorm2(self.conv2(rgb_x)))
        # input 8x44x44 -> output 8x11x11 (k=4)
        rgb_x = self.pool2(rgb_x)
        # input 8x11x11 -> output 968
        rgb_x = self.flat2(rgb_x)
        if val: print("exiting top network", rgb_x.shape)

        if val: print("entering bottom  network",env_x.shape)
        # input 21x10x10 -> output 16x8x8 (k=3)
        env_x = self.act3(self.batchnorm3(self.conv3(env_x)))
        print(env_x.shape)
        # input 16x8x8 -> output 16x6x6 (k=3)
        env_x = self.act4(self.batchnorm4(self.conv4(env_x)))
        print(env_x.shape)
        # inpput 16x6x6 -> output 576 (k=2)
        env_x = self.flat4(env_x)
        if val: print("exiting bottom netwrok" , env_x.shape)

        #if val: print(x.shape)
        # 968 + 576 = 1544
        x = torch.cat((rgb_x, env_x), dim=1)
        if val: print(x.shape)
        # input 1544 -> output 1024
        x = self.act5(self.fc5(x))
        # input 1024 -> output n_species
        x = self.fc6(x)
        return x