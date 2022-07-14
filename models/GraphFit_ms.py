import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F
import normal_estimation_utils

def _get_device(device_string):
    if device_string.lower() == 'cpu':
      return torch.device('cpu')
    if device_string.lower() == 'cuda':
      if torch.cuda.device_count() == 0:
        print("Warning: There's no GPU available on this machine!")
        return None
      return torch.device('cuda:0')
    raise Exception(
      '{} is not a valid option. Choose `cpu` or `cuda`.'.format(device_string))
def fit_Wjet(points, weights, order=2, compute_neighbor_normals=False):
    """
    Fit a "n-jet" (n-order truncated Taylor expansion) to a point clouds with weighted points.
    We assume that PCA was performed on the points beforehand.
    To do a classic jet fit input weights as a one vector.
    :param points: xyz points coordinates
    :param weights: weight vector (weight per point)
    :param order: n-order of the jet
    :param compute_neighbor_normals: bool flag to compute neighboring point normal vector

    :return: beta: polynomial coefficients
    :return: n_est: normal estimation
    :return: neighbor_normals: analytically computed normals of neighboring points
    """

    neighbor_normals = None
    batch_size, D, n_points = points.shape

    # compute the vandermonde matrix
    x = points[:, 0, :].unsqueeze(-1)# 128,128,1
    y = points[:, 1, :].unsqueeze(-1)
    z = points[:, 2, :].unsqueeze(-1)
    weights = weights.unsqueeze(-1)# 128,128,1

    # handle zero weights - if all weights are zero set them to 1

    valid_count = torch.sum(weights > 1e-3, dim=1)
    w_vector = torch.where(valid_count > 18, weights.view(batch_size, -1),
                            torch.ones_like(weights, requires_grad=True).view(batch_size, -1)).unsqueeze(-1)#128,128,1

    if order > 1:
        #pre conditioning
        h = (torch.mean(torch.abs(x), 1) + torch.mean(torch.abs(y), 1)) / 2 # 128,1 absolute value added from https://github.com/CGAL/cgal/blob/b9e320659e41c255d82642d03739150779f19575/Jet_fitting_3/include/CGAL/Monge_via_jet_fitting.h
        # h = torch.mean(torch.sqrt(x*x + y*y), dim=2)
        idx = torch.abs(h) < 0.0001
        h[idx] = 0.1
        # h = 0.1 * torch.ones(batch_size, 1, device=points.device)
        x = x / h.unsqueeze(-1).repeat(1, n_points, 1)
        y = y / h.unsqueeze(-1).repeat(1, n_points, 1)

    if order == 1:
        A = torch.cat([x, y, torch.ones_like(x)], dim=2)
    elif order == 2:
        A = torch.cat([x, y, x * x, y * y, x * y, torch.ones_like(x)], dim=2)# 128,128,6
        h_2 = h * h# 128*1
        D_inv = torch.diag_embed(1/torch.cat([h, h, h_2, h_2, h_2, torch.ones_like(h)], dim=1))
    elif order == 3:
        y_2 = y * y
        x_2 = x * x
        xy = x * y
        A = torch.cat([x, y, x_2, y_2, xy, x_2 * x, y_2 * y, x_2 * y, y_2 * x,  torch.ones_like(x)], dim=2)
        h_2 = h * h
        h_3 = h_2 * h
        D_inv = torch.diag_embed(1/torch.cat([h, h, h_2, h_2, h_2, h_3, h_3, h_3, h_3, torch.ones_like(h)], dim=1))
    elif order == 4:
        y_2 = y * y
        x_2 = x * x
        x_3 = x_2 * x
        y_3 = y_2 * y
        xy = x * y
        A = torch.cat([x, y, x_2, y_2, xy, x_3, y_3, x_2 * y, y_2 * x, x_3 * x, y_3 * y, x_3 * y, y_3 * x, y_2 * x_2,
                       torch.ones_like(x)], dim=2)
        h_2 = h * h
        h_3 = h_2 * h
        h_4 = h_3 * h
        D_inv = torch.diag_embed(1/torch.cat([h, h, h_2, h_2, h_2, h_3, h_3, h_3, h_3, h_4, h_4, h_4, h_4, h_4, torch.ones_like(h)], dim=1))
    else:
        raise ValueError("Polynomial order unsupported, please use 1 or 2 ")

    XtX = torch.matmul(A.permute(0, 2, 1),  w_vector * A)# 128,6,6
    XtY = torch.matmul(A.permute(0, 2, 1), w_vector * z)# 128,6,1
    beta = solve_linear_system(XtX, XtY, sub_batch_size=16)#128,6,1


    if order > 1: #remove preconditioning
         beta = torch.matmul(D_inv, beta)

    n_est = torch.nn.functional.normalize(torch.cat([-beta[:, 0:2].squeeze(-1), torch.ones(batch_size, 1, device=x.device, dtype=beta.dtype)], dim=1), p=2, dim=1)# 128,3

    if compute_neighbor_normals:
        beta_ = beta.squeeze().unsqueeze(1).repeat(1, n_points, 1).unsqueeze(-1)# 128,128,6,1
        if order == 1:
            neighbor_normals = n_est.unsqueeze(1).repeat(1, n_points, 1)
        elif order == 2:
            neighbor_normals = torch.nn.functional.normalize(
                torch.cat([-(beta_[:, :, 0] + 2 * beta_[:, :, 2] * x + beta_[:, :, 4] * y),
                           -(beta_[:, :, 1] + 2 * beta_[:, :, 3] * y + beta_[:, :, 4] * x),
                           torch.ones(batch_size, n_points, 1, device=x.device)], dim=2), p=2, dim=2) #128,128,3
        elif order == 3:
            neighbor_normals = torch.nn.functional.normalize(
                torch.cat([-(beta_[:, :, 0] + 2 * beta_[:, :, 2] * x + beta_[:, :, 4] * y + 3 * beta_[:, :, 5] *  x_2 +
                             2 *beta_[:, :, 7] * xy + beta_[:, :, 8] * y_2),
                           -(beta_[:, :, 1] + 2 * beta_[:, :, 3] * y + beta_[:, :, 4] * x + 3 * beta_[:, :, 6] * y_2 +
                             beta_[:, :, 7] * x_2 + 2 * beta_[:, :, 8] * xy),
                           torch.ones(batch_size, n_points, 1, device=x.device)], dim=2), p=2, dim=2)
        elif order == 4:
            # [x, y, x_2, y_2, xy, x_3, y_3, x_2 * y, y_2 * x, x_3 * x, y_3 * y, x_3 * y, y_3 * x, y_2 * x_2
            neighbor_normals = torch.nn.functional.normalize(
                torch.cat([-(beta_[:, :, 0] + 2 * beta_[:, :, 2] * x + beta_[:, :, 4] * y + 3 * beta_[:, :, 5] * x_2 +
                             2 * beta_[:, :, 7] * xy + beta_[:, :, 8] * y_2 + 4 * beta_[:, :, 9] * x_3 + 3 * beta_[:, :, 11] * x_2 * y
                             + beta_[:, :, 12] * y_3 + 2 * beta_[:, :, 13] * y_2 * x),
                           -(beta_[:, :, 1] + 2 * beta_[:, :, 3] * y + beta_[:, :, 4] * x + 3 * beta_[:, :, 6] * y_2 +
                             beta_[:, :, 7] * x_2 + 2 * beta_[:, :, 8] * xy + 4 * beta_[:, :, 10] * y_3 + beta_[:, :, 11] * x_3 +
                             3 * beta_[:, :, 12] * x * y_2 + 2 * beta_[:, :, 13] * y * x_2),
                           torch.ones(batch_size, n_points, 1, device=x.device)], dim=2), p=2, dim=2)

    return beta.squeeze(), n_est, neighbor_normals

def solve_linear_system(XtX, XtY, sub_batch_size=None):
    """
    Solve linear system of equations. use sub batches to avoid MAGMA bug
    :param XtX: matrix of the coefficients
    :param XtY: vector of the
    :param sub_batch_size: size of mini mini batch to avoid MAGMA error, if None - uses the entire batch
    :return:
    """
    if sub_batch_size is None:
        sub_batch_size = XtX.size(0)
    n_iterations = int(XtX.size(0) / sub_batch_size)
    assert sub_batch_size%sub_batch_size == 0, "batch size should be a factor of {}".format(sub_batch_size)
    beta = torch.zeros_like(XtY)
    n_elements = XtX.shape[2]
    for i in range(n_iterations):
        try:
            L = torch.cholesky(XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...], upper=False)# 16,6,6
            beta[sub_batch_size * i:sub_batch_size * (i + 1), ...] = \
                torch.cholesky_solve(XtY[sub_batch_size * i:sub_batch_size * (i + 1), ...], L, upper=False)
        except:
            # # add noise to diagonal for cases where XtX is low rank
            eps = torch.normal(torch.zeros(sub_batch_size, n_elements, device=XtX.device),
                               0.01 * torch.ones(sub_batch_size, n_elements, device=XtX.device))
            eps = torch.diag_embed(torch.abs(eps))
            XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...] = \
                XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...] + \
                eps * XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...]
            try:
                L = torch.cholesky(XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...], upper=False)
                beta[sub_batch_size * i:sub_batch_size * (i + 1), ...] = \
                    torch.cholesky_solve(XtY[sub_batch_size * i:sub_batch_size * (i + 1), ...], L, upper=False)
            except:
                beta[sub_batch_size * i:sub_batch_size * (i + 1), ...], _ =\
                    torch.solve(XtY[sub_batch_size * i:sub_batch_size * (i + 1), ...], XtX[sub_batch_size * i:sub_batch_size * (i + 1), ...])
    return beta

class GraphLayer(nn.Module):
    def __init__(self, dim=64, k1=40, k2=20):
        super(GraphLayer, self).__init__()
        self.dim = dim
        self.k1 = k1
        self.k2 = k2

        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm2d(self.dim)

        self.conv1 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn3 = nn.BatchNorm2d(self.dim)
        self.bn4 = nn.BatchNorm2d(self.dim)
        self.bn5 = nn.BatchNorm2d(self.dim)

        self.conv3 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                    self.bn3,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False),
                                    self.bn4,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x_knn1 = get_graph_feature(x, k=self.k1)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x_knn1 = self.conv1(x_knn1)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x_knn1 = self.conv2(x_knn1)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x_k1 = x_knn1.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x_knn2 = get_graph_feature(x, self.k2)
        x_knn2 = self.conv3(x_knn2)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x_knn2 = self.conv4(x_knn2)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x_k1 = x_k1.unsqueeze(-1).repeat(1, 1, 1, self.k2)

        out = torch.cat([x_knn2, x_k1], dim=1)

        out = self.conv5(out)
        out = out.max(dim=-1, keepdim=False)[0]

        return out

class PointNetFeatures(nn.Module):
    def __init__(self, num_points=500, num_scales=1, use_point_stn=False, use_feat_stn=False, point_tuple=1, sym_op='max', k1=40, k2=20):
        super(PointNetFeatures, self).__init__()
        self.num_points=num_points
        self.point_tuple=point_tuple
        self.sym_op = sym_op
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.num_scales=num_scales
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        if self.use_point_stn:
            self.stn1 = QSTN(num_scales=self.num_scales, num_points=num_points*self.point_tuple, dim=3, sym_op=self.sym_op)

        if self.use_feat_stn:
            self.stn2 = STN(num_scales=self.num_scales, num_points=num_points, dim=64, sym_op=self.sym_op)

        self.k1 = k1
        self.k2 = k2
        self.graph_layer = GraphLayer(dim=64, k1=self.k1,  k2=self.k2)


        self.ada_layer = AdaptiveLayer(C =64)


    def forward(self, x):
        n_pts = x.size()[2]
        points = x
        # input transform
        if self.use_point_stn:
            # from tuples to list of single points
            x = x.view(x.size(0), 3, -1)
            trans = self.stn1(x) # 128,3,3
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
            x = x.contiguous().view(x.size(0), 3 * self.point_tuple, -1) #(128,3,128)
            points = x
        else:
            trans = None

        x = F.relu(self.bn1(self.conv1(x)))# 128,64,128
        x = F.relu(self.bn2(self.conv2(x)))

        x_k = self.graph_layer(x)
        x = self.ada_layer([x, x_k])

        # feature transform
        if self.use_feat_stn:
            trans2 = self.stn2(x)#(128,64,64)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans2)
            x = x.transpose(2, 1)#128,64,128
        else:
            trans2 = None

        return x,  trans, trans2, points


class PointNetEncoder(nn.Module):
    def __init__(self, num_points=500, num_scales=1, use_point_stn=False, use_feat_stn=False, point_tuple=1, sym_op='max', k1=40, k2=20):
        super(PointNetEncoder, self).__init__()
        self.pointfeat = PointNetFeatures(num_points=num_points, num_scales=num_scales, use_point_stn=use_point_stn,
                         use_feat_stn=use_feat_stn, point_tuple=point_tuple, sym_op=sym_op, k1=k1, k2=k2)
        self.num_points=num_points
        self.point_tuple=point_tuple
        self.sym_op = sym_op
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.num_scales=num_scales

        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256+128+64, 1024, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(1024)

        self.k1 = k1
        self.k2 = k2

        self.graph_layer = GraphLayer(dim=128, k1=self.k1, k2=self.k2)

        self.ada_layer = AdaptiveLayer(C=128)


    def forward(self, points):
        n_pts = points.size()[2]
        pointfeat, trans, trans2, points = self.pointfeat(points)#pointfeat 128,64,128

        x1 = F.relu(self.bn2(self.conv2(pointfeat)))#(128,128,128)
        x_k = self.graph_layer(x1)
        x1 = self.ada_layer([x1, x_k])

        x2 = F.relu(self.bn3(self.conv3(x1)))  # (128,128,128)
        x = torch.cat([pointfeat, x1, x2], 1)
        x = self.bn4(self.conv4(x))#(128,1024,128)
        global_feature = torch.max(x, 2, keepdim=True)[0]#(128,1024,1)
        x = global_feature.view(-1, 1024, 1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat, x1, x2], 1), global_feature.squeeze(), trans, trans2, points


class DeepFit(nn.Module):
    def __init__(self, k=1, num_points=500, use_point_stn=False,  use_feat_stn=False, point_tuple=1,
                 sym_op='max', arch=None, n_gaussians=5, jet_order=2, weight_mode="tanh",
                 use_consistency=False, k1=40, k2=20):
        super(DeepFit, self).__init__()
        self.k = k  # k is the number of weights per point e.g. 1
        self.num_points=num_points
        self.point_tuple = point_tuple

        self.feat = PointNetEncoder(num_points=num_points, use_point_stn=use_point_stn, use_feat_stn=use_feat_stn,
                                            point_tuple=point_tuple, sym_op=sym_op, k1=k1, k2=k2)

        feature_dim = 1024 + 64 + 128 + 256

        self.conv1 = nn.Conv1d(feature_dim, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.jet_order = jet_order
        self.weight_mode = weight_mode
        self.compute_neighbor_normals = use_consistency
        self.do = torch.nn.Dropout(0.25)

        self.conv_bias = nn.Conv1d(128, 3, 1)


    def forward(self, points):

        x, global_feat, trans, trans2, points = self.feat(points)# 128,1088,128
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))#128,128,128
        bias = self.conv_bias(x)
        bias[:, :, 0] = 0 #adafit
        points = points + bias
        # point weight estimation.
        if self.weight_mode == "softmax":
            x = F.softmax(self.conv4(x))
            weights = 0.01 + x  # add epsilon for numerical robustness
        elif self.weight_mode =="tanh":
            x = torch.tanh(self.conv4(x))
            weights = (0.01 + torch.ones_like(x) + x) / 2.0  # learn the residual->weights start at 1
        elif self.weight_mode =="sigmoid":
            weights = 0.01 + torch.sigmoid(self.conv4(x)) #128,1,128

        beta, normal, neighbor_normals = fit_Wjet(points, weights.squeeze(), order=self.jet_order,
                                                              compute_neighbor_normals=self.compute_neighbor_normals)


        return normal, beta.squeeze(), weights.squeeze(), trans, trans2, neighbor_normals


class STN(nn.Module):
    def __init__(self, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(STN, self).__init__()

        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.dim*self.dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        if self.num_scales > 1:
            self.fc0 = nn.Linear(1024*self.num_scales, 1024)
            self.bn0 = nn.BatchNorm1d(1024)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            x_scales = x.new_empty(x.size(0), 1024*self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*1024:(s+1)*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.dim, dtype=x.dtype, device=x.device).view(1, self.dim*self.dim).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.dim, self.dim)
        return x


class QSTN(nn.Module):
    def __init__(self, num_scales=1, num_points=500, dim=3, sym_op='max'):
        super(QSTN, self).__init__()

        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        if self.num_scales > 1:
            self.fc0 = nn.Linear(1024*self.num_scales, 1024)
            self.bn0 = nn.BatchNorm1d(1024)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x) #128,1024,1
        else:
            x_scales = x.new_empty(x.size(0), 1024*self.num_scales, 1)
            for s in range(self.num_scales):
                x_scales[:, s*1024:(s+1)*1024, :] = self.mp1(x[:, :, s*self.num_points:(s+1)*self.num_points])
            x = x_scales

        x = x.view(-1, 1024*self.num_scales)

        if self.num_scales > 1:
            x = F.relu(self.bn0(self.fc0(x)))

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x) # 128,4

        # add identity quaternion (so the network can output 0 to leave the point cloud identical)
        iden = x.new_tensor([1, 0, 0, 0])
        x = x + iden

        # convert quaternion to rotation matrix
        x = normal_estimation_utils.batch_quat_to_rotmat(x)# 128,3,3

        return x

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx
def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


class AdaptiveLayer(nn.Module):
    def __init__(self, C, r=8):
        super(AdaptiveLayer, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(C, C // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(C // r, C, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        fea = x[0] + x[1]
        b, C, _ = fea.shape
        out = self.squeeze(fea).view(b, C)
        out = self.excitation(out).view(b, C, 1)
        attention_vectors = out.expand_as(fea)
        fea_v = attention_vectors * x[0] + (1 - attention_vectors) * x[1]
        return fea_v