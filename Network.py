import torch.nn as nn
from banzhaf import BanzhafModule
from utils import CrossEn,KL
import torch.nn.functional as F
import torch



def get_similar(view0_feat, view1_feat):
    # 归一化特征
    view0_feat = view0_feat / view0_feat.norm(dim=-1, keepdim=True)
    view1_feat = view1_feat / view1_feat.norm(dim=-1, keepdim=True)
    # 计算相似度矩阵
    retrieve_logits = torch.einsum('ad,bd->ab', [view0_feat, view1_feat])  # [N, N]
    _retrieve_logits = (retrieve_logits + retrieve_logits.T) / 2.0
    return _retrieve_logits, _retrieve_logits.T, retrieve_logits


class Encoder(nn.Module):
    def __init__(self, x_dim,z_dim):
        super(Encoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_dim,500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500,2000),
            nn.ReLU(),
            nn.Linear(2000,z_dim)
        )

    def forward(self,x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, x_dim,z_dim):
        super(Decoder,self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim,2000),
            nn.ReLU(),
            nn.Linear(2000,500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500,x_dim)
        )
    def forward(self,z):
        return self.decoder(z)

# class MLP(nn.Module):
#     def __init__(self, code_dim, layers):
#         super(MLP, self).__init__()
#         self.code_dim = code_dim
#         self.layers = layers
#         self.hidden = nn.ModuleList()
#
#         for k in range(layers):
#             linear_layer = nn.Linear(code_dim, code_dim, bias=False)
#             self.hidden.append(linear_layer)
#
#     def forward(self, z):
#         for l in self.hidden:
#             z = l(z)
#         return z

class Confounder(nn.Module):
    def __init__(self, hid_dim, bias_dim_factor):
        super().__init__()
        low_dim = int(hid_dim / bias_dim_factor)
        self.encoder = nn.Sequential(
            nn.Linear(hid_dim, low_dim),
            nn.ReLU(),
            nn.LayerNorm(low_dim,eps=1e-12)
        )
        self.remap = nn.Linear(low_dim, hid_dim)

    def forward(self,R_v):
        R_v_detached = R_v.detach()
        z_c_low = self.encoder(R_v_detached)
        z_c_high = self.remap(z_c_low)
        return z_c_low, z_c_high


class Network(nn.Module):
    def __init__(self,args, dim_list):
        super(Network,self).__init__()
        self.args = args
        self._latent_dim = args.n_z
        self.V = len(dim_list)
        self.class_dim =args.num_cluster
        self.bias_dim_factor = args.bias_dim_factor
        self.num_cluster = args.num_cluster

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.confounder = nn.ModuleList()
        self.cluster = nn.Sequential(
            nn.Linear(self._latent_dim, self.num_cluster),
            nn.Softmax(dim=1)
        )

        for v in range(self.V):
            self.encoders.append(Encoder(dim_list[v],self._latent_dim))
            self.decoders.append(Decoder(dim_list[v],self._latent_dim))
            self.confounder.append(Confounder(self._latent_dim,self.bias_dim_factor))




    def forward(self,data):
        out = {
            'x_rec':[],
            'feature':[],
            'biases': [],
            'bias_only_logits':[],
            'biases_low_dim':[],
            'debiased_featrue': [],
        }
        for v in range(self.V):
            z_v = self.encoders[v](data[v])
            recon_x = self.decoders[v](z_v)
            out['x_rec'].append(recon_x)
            out['feature'].append(z_v)
            R_v = z_v.detach().clone()
            z_c_low, z_c_high = self.confounder[v](R_v)
            out['biases'].append(z_c_high)
            out['biases_low_dim'].append(z_c_low)

            z_debiased = z_v-z_c_high
            out['debiased_featrue'].append(z_debiased)
            bias_only_logits = self.cluster(z_debiased)
            out['bias_only_logits'].append(bias_only_logits)
        z_both = torch.stack(out['debiased_featrue'], dim=0).mean(dim=0)
        # z_both = torch.cat(out['debiased_featrue'],dim=1)
        bia_logits = self.cluster(z_both)
        out['bia_logits'] = bia_logits
        out['consensus_feature'] = z_both


        return out
