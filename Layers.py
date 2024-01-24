import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, inputchannel,outputchannel):
        super(MLP, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=inputchannel,out_channels=outputchannel,kernel_size=1)
        self.bn1 = nn.BatchNorm1d(outputchannel)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return x

class DPIB(nn.Module):
    def __init__(self, channel):
        super(DPIB, self).__init__()

        self.MLPA1 = MLP(inputchannel=channel, outputchannel=int(channel / 2))
        self.MLPA2 = MLP(inputchannel=int(channel * 3 / 2), outputchannel=int(channel / 2))
        self.MLPA3 = MLP(inputchannel=channel+2*int(channel / 2), outputchannel=int(channel / 2))

        self.MLPB1 = MLP(inputchannel=channel, outputchannel=int(channel / 2))
        self.MLPB2 = MLP(inputchannel=int(channel * 3 / 2), outputchannel=int(channel / 2))
        self.MLPB3 = MLP(inputchannel=channel+2*int(channel / 2), outputchannel=channel-int(channel / 2))

    def forward(self, x):
        A1out=self.MLPA1(x)
        A2out=self.MLPA2(torch.cat((x,A1out),dim=1))
        A3out=self.MLPA3(torch.cat((x,A1out,A2out),dim=1))

        B1out = self.MLPB1(x)
        B2out = self.MLPB2(torch.cat((x, B1out), dim=1))
        B3out = self.MLPB3(torch.cat((x, B1out, B2out), dim=1))

        out=torch.cat((A3out,B3out),dim=1)+x
        return out


def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:shape=(B, M, N)
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) #  Important for dist = 0.
    return torch.sqrt(dists).float()

def fps(xyz, M):
    '''
    Sample M points from points according to farthest point sampling (FPS) algorithm.
    :param xyz: shape=(B, N, 3)
    :return: inds: shape=(B, M)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)
    dists = torch.ones(B, N).to(device) * 1e5
    inds = torch.randint(0, N, size=(B, ), dtype=torch.long).to(device)
    batchlists = torch.arange(0, B, dtype=torch.long).to(device)
    for i in range(M):
        centroids[:, i] = inds
        cur_point = xyz[batchlists, inds, :] # (B, 3)
        cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), xyz))
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]
        inds = torch.max(dists, dim=1)[1]
    return centroids

def gather_points(points, inds):
    '''
    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]

def ball_query(xyz, new_xyz, radius, K):
    '''
    :param xyz: shape=(B, N, 3)
    :param new_xyz: shape=(B, M, 3)
    :param radius: int
    :param K: int, an upper limit samples
    :return: shape=(B, M, 3)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]
    grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)#(B,M,N)
    dists = get_dists(new_xyz, xyz)#(B,M,N)
    grouped_inds[dists > radius] = N
    grouped_inds = torch.sort(grouped_inds, dim=-1)[0][:, :, :K]#(B,M,C)
    grouped_min_inds = grouped_inds[:, :, 0:1].repeat(1, 1, K)
    grouped_inds[grouped_inds == N] = grouped_min_inds[grouped_inds == N]
    return grouped_inds

class PointConv(nn.Module):
    def __init__(self,Nout,featurein,featureout):
        super(PointConv,self).__init__()
        self.Nout=Nout
        self.featurein=featurein
        self.featureout=featureout
        self.para1=nn.Parameter(torch.rand(featurein,featureout))
        self.para2 = nn.Parameter(torch.rand(featurein, featureout))
        self.para3 = nn.Parameter(torch.rand(featurein, featureout))
        self.acti=nn.ReLU()
    def forward(self,rgb,xyz,R1=0.5,R2=0.5,K1=3,K2=2):
        '''
        :param rgb: shape=(B, featurenum, N)
        :param xyz: shape=(B, N, 3)
        :param R1,R2:float,
        :param K1,K2: int, upper limit samples
        :return: shape=(B,featurenum/2 , N/2)
        '''
        B, featurenum, Nin=rgb.shape
        fi_id=fps(xyz,self.Nout)#(B,N/2)
        fi_xyz=gather_points(xyz,fi_id)#(B,N/2,3)
        fij_id=ball_query(xyz,fi_xyz,R1,K1+1)
        fi1_id=fij_id[:,:,1]
        fi2_id=fij_id[:,:,2]
        fi3_id=fij_id[:,:,3]#B,N/2

        sub1 = gather_points(xyz, fi1_id)
        sub2 = gather_points(xyz, fi2_id)
        sub3 = gather_points(xyz, fi3_id)#B,N/2,3

        subcloud1_id = ball_query(xyz, sub1, R2, K2+1)
        subcloud2_id = ball_query(xyz, sub2, R2, K2+1)
        subcloud3_id = ball_query(xyz, sub3, R2, K2+1)#B,N/2,K2+1

        subcloud1=gather_points(torch.transpose(rgb,1,2),subcloud1_id)
        subcloud2 = gather_points(torch.transpose(rgb, 1, 2), subcloud2_id)
        subcloud3 = gather_points(torch.transpose(rgb, 1, 2), subcloud3_id)#B,N/2,K2+1,featurenum

        y1=torch.matmul(subcloud1,self.para1)
        y1=y1[:,:,0:1,:]+y1[:,:,1:2,:]+y1[:,:,2:3,:]
        y1=self.acti(y1)

        y2=torch.matmul(subcloud2,self.para2)
        y2=y2[:,:,0:1,:]+y2[:,:,1:2,:]+y2[:,:,2:3,:]
        y2=self.acti(y2)

        y3=torch.matmul(subcloud3,self.para3)
        y3=y3[:,:,0:1,:]+y3[:,:,1:2,:]+y3[:,:,2:3,:]
        y3=self.acti(y3)

        y=torch.cat((y1,y2,y3),dim=2)
        y=torch.max_pool2d(input=y,kernel_size=(3,1))
        y=y.squeeze()
        y=torch.transpose(y,1,2)
        return y,fi_xyz

class PointDeconv(nn.Module):
    def __init__(self,Nout,featurein,featureout):
        super(PointDeconv, self).__init__()
        self.Nout = Nout
        self.featurein = featurein
        self.featureout = featureout

        self.mlp1 = MLP(inputchannel=featurein, outputchannel=featureout)
        self.mlp2 = MLP(inputchannel=featureout, outputchannel=featureout)
        self.mlp3 = MLP(inputchannel=featureout, outputchannel=featureout)

    def forward(self, rgb, xyzin,xyzout, R1=0.5, R2=0.5, K1=3, K2=2):
        # rgb:(featurein,Nin) xyzout:(Nout,3)
        rgb = self.mlp1(rgb)
        rgb = self.mlp2(rgb)
        rgb = self.mlp3(rgb)
        distance=get_dists(xyzout,xyzin)
        distance=torch.square(distance)
        distance=torch.sort(input=distance,dim=-1)

        weight_value=1/distance.values[:,:,0:3]
        weightsum=torch.sum(input=weight_value,dim=-1,keepdim=True)
        weightsum=torch.repeat_interleave(weightsum,3,dim=-1)
        weight_value=torch.div(weight_value,weightsum)#B,M,3
        weight_value=torch.unsqueeze(weight_value,dim=2)#B,M,1,3

        dist_ind=distance.indices[:,:,0:3]#B,M,K
        rgb_=gather_points(torch.transpose(rgb, 1, 2),dist_ind)#B,M,3,C

        newrgb_=torch.matmul(weight_value,rgb_)#B,M,1,C
        newrgb_=torch.squeeze(newrgb_)#B,M,C
        newrgb_=torch.transpose(newrgb_, 1, 2)#B,C,M

        return newrgb_