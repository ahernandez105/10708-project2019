import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class CEN_RCN_Simple(nn.Module):
    def __init__(self, context_encoder, encoding_dim, pyz, mask_u=False,
                pyz_learnable=False):
        super(CEN_RCN_Simple, self).__init__()

        self.context_encoder = context_encoder

        self.mask_u = mask_u

        # TODO: change pyz to log pyz
        print(f"pyz_learnable {pyz_learnable}")
        # self.log_pyz = nn.Parameter(pyz.log(), requires_grad=pyz_learnable)
        self.pyz = nn.Parameter(pyz, requires_grad=pyz_learnable)

    def forward(self, context, S):
        # dot-product attention between rule representations and
        # fused rep of context and attributes

        # input x: n_train x n_features
        # x_rep: n_train x n_hidden
        # S: n_rules x n_train
        # S_rep: n_rules x n_hidden
        # h: n_train x n_hidden

        phi = self.context_encoder(context)
        S_rep = torch.matmul(S.transpose(0, 1), phi)

        # u: n_train x n_rules
        # scores for each
        u = torch.matmul(phi, S_rep.transpose(0, 1))

        if self.mask_u:
            # only allow rules that the datapoint satisfies. We use S to mask
            # assumes that each datapoint satisfies at least one antecedent

            # print(S.sum(-1))
            # if not S.sum(-1).min() > 0:
            #     print("no antecedents satisfy datapoint!")
            #     raise ValueError

            if not S.sum(-1).min() > 0:
                pz = F.softmax(torch.ones_like(u), dim=-1)
            else:
                ninf = torch.full_like(u, np.NINF)
                masked_u = torch.where(S > 0, u, ninf)
                pz = F.softmax(masked_u, dim=-1)
            # pz = F.log_softmax(masked_u, dim=-1)
        
        else:
            pz = F.softmax(u, dim=-1)
            # pz = F.log_softmax(u, dim=-1)

        # sum_py = self.log_pyz[None, :, :] + pz[:, :, None]
        # py = sum_py.logsumexp(1)

        # TODO: add sequence transition parameters
        py = torch.matmul(pz, self.pyz)

        return pz, py

class CEN_RCN(nn.Module):
    def __init__(self, context_encoder, encoding_dim, n_features, n_hidden, pyz, mask_u=False,
                pyz_learnable=False):
        super(CEN_RCN, self).__init__()

        self.context_encoder = context_encoder

        self.fusion = nn.Linear(encoding_dim + n_features, n_hidden)
        self.input_encoder = nn.Linear(n_features, n_hidden)
        # self.fusion = nn.Sequential(
        #     nn.Linear(encoding_dim + n_features, n_hidden),
        #     nn.ReLU(True),
        #     nn.Linear(n_hidden, n_hidden)
        # )
        # self.input_encoder = nn.Sequential(
        #     nn.Linear(n_features, n_hidden),
        #     nn.ReLU(True),
        #     nn.Linear(n_hidden, n_hidden)
        # )

        self.mask_u = mask_u

        # TODO: change pyz to log pyz
        print(f"pyz_learnable {pyz_learnable}")
        # self.log_pyz = nn.Parameter(pyz.log(), requires_grad=pyz_learnable)
        self.pyz = nn.Parameter(pyz, requires_grad=pyz_learnable)
        
    def forward(self, context, x, S):
        # dot-product attention between rule representations and
        # fused rep of context and attributes

        # input x: n_train x n_features
        # x_rep: n_train x n_hidden
        # S: n_rules x n_train
        # S_rep: n_rules x n_hidden
        # h: n_train x n_hidden

        phi = self.context_encoder(context)
        h = self.fusion(torch.cat([phi, x], dim=-1))

        x_rep = self.input_encoder(x)
        # x_rep = phi
        S_rep = torch.matmul(S.transpose(0, 1), x_rep)

        # u: n_train x n_rules
        # scores for each
        u = torch.matmul(h, S_rep.transpose(0, 1))

        if self.mask_u:
            # only allow rules that the datapoint satisfies. We use S to mask
            # assumes that each datapoint satisfies at least one antecedent

            # print(S.sum(-1))
            # if not S.sum(-1).min() > 0:
            #     print("no antecedents satisfy datapoint!")
            #     raise ValueError

            if not S.sum(-1).min() > 0:
                pz = F.softmax(torch.ones_like(u), dim=-1)
            else:
                ninf = torch.full_like(u, np.NINF)
                masked_u = torch.where(S > 0, u, ninf)
                pz = F.softmax(masked_u, dim=-1)
            # pz = F.log_softmax(masked_u, dim=-1)
        
        else:
            pz = F.softmax(u, dim=-1)
            # pz = F.log_softmax(u, dim=-1)

        # sum_py = self.log_pyz[None, :, :] + pz[:, :, None]
        # py = sum_py.logsumexp(1)

        # TODO: add sequence transition parameters
        py = torch.matmul(pz, self.pyz)

        return pz, py

class FF(nn.Module):
    def __init__(self, context_encoder, encoding_dim, output_dim):
        """
        Directly predict p(y|x)
        """
        super(FF, self).__init__()

        self.context_encoder = context_encoder

        self.final = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(encoding_dim, output_dim)
        )

    def forward(self, x):
        phi = self.context_encoder(x)
        return F.softmax(self.final(phi), dim=-1)
