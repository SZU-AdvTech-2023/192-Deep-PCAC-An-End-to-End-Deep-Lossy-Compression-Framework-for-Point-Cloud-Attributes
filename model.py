from Model import *
import torch.nn as nn

class AttributeEncoder(nn.Module):
    def __init__(self,Nin,Ns):
        super(AttributeEncoder, self).__init__()
        self.mlp1 = MLP(inputchannel=3, outputchannel=16)
        self.mlp2 = MLP(inputchannel=16, outputchannel=64)
        self.mlp3 = MLP(inputchannel=64, outputchannel=128)
        self.mlp4 = MLP(inputchannel=128, outputchannel=256)
        self.mlp5 = MLP(inputchannel=256, outputchannel=512)
        self.dpib1 = DPIB(channel=16)
        self.dpib2 = DPIB(channel=64)
        self.dpib3 = DPIB(channel=128)
        self.dpib4 = DPIB(channel=256)
        self.dpib5 = DPIB(channel=512)

        self.PointConv1=PointConv(Nout=int(Nin/2),featurein=512,featureout=256)
        self.dpib6 = DPIB(channel=256)

        self.PointConv2 = PointConv(Nout=int(Nin / 4), featurein=256, featureout=128)
        self.dpib7 = DPIB(channel=128)

        self.PointConv3 = PointConv(Nout=Ns, featurein=128, featureout=64)
        self.dpib8 = DPIB(channel=64)

        self.mlp6 = MLP(inputchannel=64, outputchannel=8)


    def forward(self,rgb,geometry):
        x = self.dpib1(self.mlp1(rgb))
        x = self.dpib2(self.mlp2(x))
        x = self.dpib3(self.mlp3(x))
        x = self.dpib4(self.mlp4(x))
        x = self.dpib5(self.mlp5(x))
        x,geometry1 = self.PointConv1(x,geometry)
        x = self.dpib6(x)
        x, geometry2 = self.PointConv2(x, geometry1)
        x = self.dpib7(x)
        x, geometry3 = self.PointConv3(x, geometry2)
        x = self.dpib8(x)
        x=self.mlp6(x)
        return x,geometry3

def GeometrySample(geometry,N,Ns):
    ls=[]
    ls.append(geometry)

    fi_id = fps(geometry, int(N/2))  # (B,N/2)
    fi_xyz = gather_points(geometry, fi_id)  # (B,N/2,3)
    ls.append(fi_xyz)

    fi_id = fps(fi_xyz, int(N / 4))  # (B,N/2)
    fi_xyz = gather_points(geometry, fi_id)  # (B,N/2,3)
    ls.append(fi_xyz)

    fi_id = fps(fi_xyz, Ns)  # (B,N/2)
    fi_xyz = gather_points(geometry, fi_id)  # (B,N/2,3)
    ls.append(fi_xyz)

    return ls

class AttributeDecoder(nn.Module):
    def __init__(self,Nout,Ns):
        super(AttributeDecoder, self).__init__()
        self.Nout=Nout
        self.Ns=Ns
        self.mlp1 = MLP(inputchannel=8, outputchannel=64)

        self.PointDec1=PointDeconv(Nout=int(Nout/4),featurein=64,featureout=128)
        self.dpib1=DPIB(channel=128)
        self.PointDec2 = PointDeconv(Nout=int(Nout / 2), featurein=128, featureout=256)
        self.dpib2=DPIB(channel=256)
        self.PointDec3 = PointDeconv(Nout=int(Nout / 1), featurein=256, featureout=512)
        self.dpib3 = DPIB(channel=512)

        self.mlp4 = MLP(inputchannel=512, outputchannel=256)
        self.dpib4 = DPIB(channel=256)
        self.mlp5 = MLP(inputchannel=256, outputchannel=128)
        self.dpib5 = DPIB(channel=128)
        self.mlp6 = MLP(inputchannel=128, outputchannel=64)
        self.dpib6 = DPIB(channel=64)
        self.mlp7 = MLP(inputchannel=64, outputchannel=16)
        self.dpib7 = DPIB(channel=16)
        self.mlp8 = MLP(inputchannel=16, outputchannel=3)
        self.dpib8 = DPIB(channel=3)

    def forward(self,latentcode,geometry):
        geolist=GeometrySample(geometry,N=self.Nout,Ns=self.Ns)

        x = self.mlp1(latentcode)
        x = self.dpib1(self.PointDec1(x, geolist[3], geolist[2]))
        x = self.dpib2(self.PointDec2(x, geolist[2], geolist[1]))
        x = self.dpib3(self.PointDec3(x, geolist[1], geolist[0]))

        x = self.dpib4(self.mlp4(x))
        x = self.dpib5(self.mlp5(x))
        x = self.dpib6(self.mlp6(x))
        x = self.dpib7(self.mlp7(x))
        x = self.dpib8(self.mlp8(x))

        return x

class HyperEncoder(nn.Module):
    def __init__(self,Ns):
        super(HyperEncoder, self).__init__()
        self.PointConv1=PointConv(Nout=int(Ns/2),featurein=8,featureout=8)
        self.PointConv2 = PointConv(Nout=int(Ns / 4), featurein=8, featureout=8)
        self.mlp1=MLP(inputchannel=8,outputchannel=6)
        self.mlp2 = MLP(inputchannel=6, outputchannel=6)
        self.mlp3 = MLP(inputchannel=6, outputchannel=6)
    def forward(self,x,geoin):

        x,geo=self.PointConv1(x,geoin)
        x,geo_=self.PointConv2(x,geo)
        x=self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        return x,geo_

class HyperDecoder(nn.Module):
    def __init__(self,Ns):
        super(HyperDecoder, self).__init__()
        self.Ns=Ns
        self.PointDec1=PointDeconv(Nout=int(Ns/2),featurein=6,featureout=12)
        self.PointDec2 = PointDeconv(Nout=Ns, featurein=12, featureout=12)
        self.mlp=MLP(inputchannel=12,outputchannel=24)
        self.mlp1 = MLP(inputchannel=24, outputchannel=8)
        self.mlp2 = MLP(inputchannel=24, outputchannel=8)
    def forward(self,x,geo):
        ls=GeometrySample(geometry=geo,N=self.Ns,Ns=self.Ns)
        x=self.PointDec1(x,ls[2],ls[1])
        x=self.PointDec2(x,ls[1],ls[0])
        x=self.mlp(x)
        ave=self.mlp1(x)
        std=self.mlp2(x)
        return ave,std