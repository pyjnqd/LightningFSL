import torch
import torch.nn as nn
import torch.nn.functional as F
from qpth.qp import QPFunction
import cv2
from tqdm import tqdm
import math
class DeepEMD(nn.Module):
    def __init__(self, solver = "opencv", temperature = 12.5):
        super().__init__()
        self.solver = solver
        self.temperature = temperature

    def forward(self, query, proto, way, shot, return_sim_map = False):
        #proto:[B,M,c,h,w]
        #query:[B,N,c,h,w]

        #weight_1是query的权重
        weight_1 = self.get_weight_vector(query, proto)
        weight_2 = self.get_weight_vector(proto, query)

        similarity_map = self.get_similiarity_map(proto, query)
        if self.solver == 'opencv':
            logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='opencv')
        else:
            logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='qpth')

        return logits



    def get_weight_vector(self, A, B):
        '''
        A:[B,M,c,h,w]
        B:[B,N,c,h,w]
        A对B中所有图片两两求权值，结果为(M,N,h*w)的张量
        '''

        M = A.shape[1]
        N = B.shape[1]

        B = B.reshape([-1] + list(B.shape[-3:]))
        B = F.normalize(B, p=2, dim=1, eps=1e-12)
        A = F.normalize(A, p=2, dim=2, eps=1e-12)

        B = F.adaptive_avg_pool2d(B, [1, 1])

        B = B.reshape([-1,N]+list(B.shape[-3:]))
        B = B.repeat(1, 1, 1, A.shape[3], A.shape[4])

        A = A.unsqueeze(2)
        B = B.unsqueeze(1)
        #下面操作两两求权值
        A = A.repeat(1, 1, N, 1, 1, 1)
        B = B.repeat(1, M, 1, 1, 1, 1)
        combination = (A * B).sum(3)#在channel维度求内积
        combination = combination.view(A.size(0),M, N, -1)
        combination = F.relu(combination) + 1e-3#保证权值是正数

        return combination

    def get_emd_distance(self, similarity_map, weight_1, weight_2, solver='opencv'):
        """
        similarity_map:[bs,num_query, way,h*w,h*w]
        weight_1:[bs,num_query, way, h*w]
        weight_2:[bs, way, num_query, h*w]
        return:[bs, num_query, num_proto]
        """
        bs = similarity_map.shape[0]
        num_query = similarity_map.shape[1]
        num_proto = similarity_map.shape[2]
        num_node=weight_1.shape[-1]#共多少个position

        if solver == 'opencv':  # use openCV solver
            for k in range(bs):
                for i in range(num_query):
                    for j in range(num_proto):
                        _, flow = emd_inference_opencv(1 - similarity_map[k,i, j, :, :], weight_1[k,i, j, :], weight_2[k,j, i, :])

                        similarity_map[k, i, j, :, :] =(similarity_map[k, i, j, :, :])*torch.from_numpy(flow).cuda()

            temperature=(self.temperature/num_node)
            logitis = similarity_map.sum(-1).sum(-1) *  temperature
            return logitis

        elif solver == 'qpth':
            weight_2 = weight_2.permute(0, 2, 1, 3)
            similarity_map = similarity_map.view(bs, num_query * num_proto, similarity_map.shape[-2],
                                                 similarity_map.shape[-1])
            weight_1 = weight_1.view(bs, num_query * num_proto, weight_1.shape[-1])
            weight_2 = weight_2.reshape(bs, num_query * num_proto, weight_2.shape[-1])

            _, flows = emd_inference_qpth(1 - similarity_map, weight_1, weight_2,form='L2', l2_strength=0.000001)

            logitis=(flows*similarity_map).view(bs, num_query, num_proto,flows.shape[-2],flows.shape[-1])
            temperature = (self.temperature / num_node)
            logitis = logitis.sum(-1).sum(-1) *  temperature
        else:
            raise ValueError('Unknown Solver')

        return logitis


    def get_similiarity_map(self, proto, query):

        way = proto.shape[1]
        num_query = query.shape[1]
        bs = proto.shape[0]
        query = query.view(query.shape[0], query.shape[1], query.shape[2],-1)
        proto = proto.view(proto.shape[0], proto.shape[1], proto.shape[2],-1)

        proto = proto.unsqueeze(1).repeat([1, num_query, 1, 1, 1])
        query = query.unsqueeze(2).repeat([1 ,1, way, 1, 1])#两两求dense cosine距离
        proto = proto.permute(0, 1, 2, 4, 3)
        query = query.permute(0, 1, 2, 4, 3)#把channel维度放到最后
        feature_size = proto.shape[-2]
        #到这里的张量长这样：[num_query, way, h*w, dim]
        proto = proto.unsqueeze(-3)#[num_query, way, 1, h*w, dim]
        query = query.unsqueeze(-2)#这两个操作让feature维度也pairwise
        query = query.repeat(1, 1, 1, 1, feature_size, 1)#[num_query, way,h*w,h*w,dim]
        similarity_map = F.cosine_similarity(proto, query, dim=-1)#broadcast proto,get[num_query, way,h*w,h*w]

        return similarity_map



def create_model():
    return DeepEMD()

def emd_inference_qpth(distance_matrix, weight1, weight2, form='QP', l2_strength=0.0001):
    """
    to use the QP solver QPTH to derive EMD (LP problem),
    one can transform the LP problem to QP,
    or omit the QP term by multiplying it with a small value,i.e. l2_strngth.
    :param distance_matrix: [bs, num_query * num_proto, h*w, h*w]
    :param weight1: [bs, num_query * num_proto, h*w]
    :param weight2: [bs, num_query * num_proto, h*w]
    :return:
    emd distance: nbatch*1
    flow : nbatch * weight_number *weight_number

    """

    weight1 = (weight1 * weight1.shape[-1]) / weight1.sum(2).unsqueeze(2)
    weight2 = (weight2 * weight2.shape[-1]) / weight2.sum(2).unsqueeze(2)
    bs = distance_matrix.shape[0]
    nbatch = distance_matrix.shape[1]
    nelement_distmatrix = distance_matrix.shape[2] * distance_matrix.shape[3]
    nelement_weight1 = weight1.shape[2]
    nelement_weight2 = weight2.shape[2]

    Q_1 = distance_matrix.view(bs,-1, 1, nelement_distmatrix).double()

    if form == 'QP':
        # version: QTQ
        Q = torch.bmm(Q_1.transpose(3, 2), Q_1).double().cuda() + 1e-4 * torch.eye(
            nelement_distmatrix).double().cuda().unsqueeze(0).unsqueeze(0).repeat(bs,nbatch, 1, 1)  # 0.00001 *
        p = torch.zeros(bs,nbatch, nelement_distmatrix).double().cuda()
    elif form == 'L2':
        # version: regularizer
        Q = (l2_strength * torch.eye(nelement_distmatrix).double()).cuda().unsqueeze(0).unsqueeze(0).repeat(bs,nbatch, 1, 1)
        p = distance_matrix.view(bs,nbatch, nelement_distmatrix).double()
    else:
        raise ValueError('Unkown form')

    h_1 = torch.zeros(bs,nbatch, nelement_distmatrix).double().cuda()
    h_2 = torch.cat([weight1, weight2], 2).double()
    h = torch.cat((h_1, h_2), 2)

    G_1 = -torch.eye(nelement_distmatrix).double().cuda().unsqueeze(0).unsqueeze(0).repeat(bs,nbatch, 1, 1)
    G_2 = torch.zeros([bs,nbatch, nelement_weight1 + nelement_weight2, nelement_distmatrix]).double().cuda()
    # sum_j(xij) = si
    for i in range(nelement_weight1):
        G_2[:,:, i, nelement_weight2 * i:nelement_weight2 * (i + 1)] = 1
    # sum_i(xij) = dj
    for j in range(nelement_weight2):
        G_2[:,:, nelement_weight1 + j, j::nelement_weight2] = 1
    #xij>=0, sum_j(xij) <= si,sum_i(xij) <= dj, sum_ij(x_ij) = min(sum(si), sum(dj))
    G = torch.cat((G_1, G_2), 2)
    A = torch.ones(bs,nbatch, 1, nelement_distmatrix).double().cuda()
    b = torch.min(torch.sum(weight1, 2), torch.sum(weight2, 2)).unsqueeze(2).double()
    flow = torch.zeros(bs,nbatch,nelement_weight1*nelement_weight1).double().cuda()
    for i in range(bs):
        flow[i] = QPFunction(verbose=-1)(Q[i], p[i], G[i], h[i], A[i], b[i])

    emd_score = torch.sum((1 - Q_1).squeeze() * flow, 2)
    return emd_score, flow.view(bs,-1, nelement_weight1, nelement_weight2)


def emd_inference_opencv(cost_matrix, weight1, weight2):
    # cost matrix is a tensor of shape [N,N]
    cost_matrix = cost_matrix.detach().cpu().numpy()

    weight1 = F.relu(weight1) + 1e-5
    weight2 = F.relu(weight2) + 1e-5

    weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
    weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()

    cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
    return cost, flow

def emd_inference_opencv_test(distance_matrix,weight1,weight2):
    distance_list = []
    flow_list = []

    for i in range (distance_matrix.shape[0]):
        cost,flow=emd_inference_opencv(distance_matrix[i],weight1[i],weight2[i])
        distance_list.append(cost)
        flow_list.append(torch.from_numpy(flow))

    emd_distance = torch.Tensor(distance_list).cuda().double()
    flow = torch.stack(flow_list, dim=0).cuda().double()

    return emd_distance,flow