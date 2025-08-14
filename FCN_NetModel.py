
import scipy.misc as misc
import torch
import copy
import torchvision.models as models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):

    def __init__(self, CatDict):
            super(Net, self).__init__()
            self.Encoder = models.resnet101(pretrained=True)
            self.PSPScales = [1, 1 / 2, 1 / 4, 1 / 8]

            self.PSPLayers = nn.ModuleList() 
            for Ps in self.PSPScales:
                self.PSPLayers.append(nn.Sequential(
                    nn.Conv2d(2048, 1024, stride=1, kernel_size=3, padding=1, bias=True)))
            self.PSPSqueeze = nn.Sequential(
                nn.Conv2d(4096, 512, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )
            
            self.SkipConnections = nn.ModuleList()
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(1024, 512, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()))
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(512, 256, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))
            self.SkipConnections.append(nn.Sequential(
                nn.Conv2d(256, 256, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))

            self.SqueezeUpsample = nn.ModuleList()
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(1024, 512, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()))
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(256 + 512, 256, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))
            self.SqueezeUpsample.append(nn.Sequential(
                nn.Conv2d(256 + 256, 256, stride=1, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()))

            self.OutLayersList =nn.ModuleList()
            self.OutLayersDict={}
            for f,nm in enumerate(CatDict):

                    self.OutLayersDict[nm]= nn.Conv2d(256, 2, stride=1, kernel_size=3, padding=1, bias=False)
                    self.OutLayersList.append(self.OutLayersDict[nm])

    def forward(self,Images,UseGPU=True,TrainMode=True, FreezeBatchNormStatistics=False):

                RGBMean = [123.68,116.779,103.939]
                RGBStd = [65,65,65]
                if TrainMode:
                        tp=torch.FloatTensor
                else:
                    self.half()
                    tp=torch.HalfTensor
                InpImages = torch.autograd.Variable(torch.from_numpy(Images.astype(float)), requires_grad=False).transpose(2,3).transpose(1, 2).type(tp)
                if FreezeBatchNormStatistics==True: self.eval()
                if UseGPU:
                    InpImages=InpImages.cuda()
                    self.cuda()
                else:
                    self=self.cpu()
                    self.float()
                    InpImages=InpImages.type(torch.float).cpu()
                for i in range(len(RGBMean)): InpImages[:, i, :, :]=(InpImages[:, i, :, :]-RGBMean[i])/RGBStd[i]
                x=InpImages
                SkipConFeatures=[] 

                x = self.Encoder.conv1(x)
                x = self.Encoder.bn1(x)

                x = self.Encoder.relu(x)
                x = self.Encoder.maxpool(x)
                x = self.Encoder.layer1(x)
                SkipConFeatures.append(x)
                x = self.Encoder.layer2(x)
                SkipConFeatures.append(x)
                x = self.Encoder.layer3(x)
                SkipConFeatures.append(x)
                x = self.Encoder.layer4(x)

                PSPSize=(x.shape[2],x.shape[3])

                PSPFeatures=[] 
                for i,PSPLayer in enumerate(self.PSPLayers): 
                      NewSize=(np.array(PSPSize)*self.PSPScales[i]).astype(int)
                      if NewSize[0] < 1: NewSize[0] = 1
                      if NewSize[1] < 1: NewSize[1] = 1

                      y = nn.functional.interpolate(x, tuple(NewSize), mode='bilinear')
                      y = PSPLayer(y)
                      y = nn.functional.interpolate(y, PSPSize, mode='bilinear')

                      PSPFeatures.append(y)
                x=torch.cat(PSPFeatures,dim=1)
                x=self.PSPSqueeze(x)

                for i in range(len(self.SkipConnections)):
                  sp=(SkipConFeatures[-1-i].shape[2],SkipConFeatures[-1-i].shape[3])
                  x=nn.functional.interpolate(x,size=sp,mode='bilinear') #Resize
                  x = torch.cat((self.SkipConnections[i](SkipConFeatures[-1-i]),x), dim=1)
                  x = self.SqueezeUpsample[i](x)

                self.OutProbDict = {}
                self.OutLbDict = {}

                for nm in self.OutLayersDict:

                    l=self.OutLayersDict[nm](x)
                    l = nn.functional.interpolate(l,size=InpImages.shape[2:4],mode='bilinear')
                    Prob = F.softmax(l, dim=1)
                    tt, Labels = l.max(1) 
                    self.OutProbDict[nm]=Prob
                    self.OutLbDict[nm] = Labels
                return  self.OutProbDict,self.OutLbDict








