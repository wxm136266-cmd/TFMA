import torch
import torch.nn as nn
import torch.nn.functional as F
import functools 
import numpy as np
import torchvision.models as models
# -------------------Feature Extraction----------------------------------------------------------------

class LKA(nn.Module):
    def __init__(self, dim, kernel_size, dilated_rate=3):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size, padding='same', groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding='same', groups=dim, dilation=dilated_rate)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.norm = nn.BatchNorm2d(dim)
    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)

        return u*attn

class my_norm(nn.Module):
    def __init__(self, shape=4096):
        super().__init__()
        self.shape = shape
        self.norm = nn.LayerNorm(shape)
    def forward(self, x):
        B,C,H,W = x.shape
        x = x.view(B,C,-1)
        x = self.norm(x)
        x = x.view(B,C,H,W)
        return x

class MultiScaleExtractor(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        # self.head_pw = nn.Conv2d(dim, dim, 1)
        self.tail_pw = nn.Conv2d(dim, dim, 1)

        self.LKA3 = LKA(dim, kernel_size=3)
        self.LKA5 = LKA(dim, kernel_size=5)
        self.LKA7 = LKA(dim, kernel_size=7)
        self.norm3 = nn.BatchNorm2d(dim)
        self.norm5 = nn.BatchNorm2d(dim)
        self.norm7 = nn.BatchNorm2d(dim)

        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.conv_cn = nn.Conv2d(dim, dim, 3, groups=dim,padding=1)
        self.norm_last = nn.BatchNorm2d(dim)
    def forward(self, x):
        x_copy = x.clone()
        # x = self.head_pw(x)

        x3 = self.LKA3(x) + x
        x3 = self.norm3(x3)
        x5 = self.LKA5(x) + x
        x5 = self.norm5(x5)
        x7 = self.LKA7(x) + x
        x7 = self.norm7(x7)

        x = F.gelu(x3 + x5 + x7)
        x = self.tail_pw(x) + x_copy

        x = self.pointwise(x)
        x = self.conv_cn(x)
        x = F.gelu(self.norm_last(x))
        return x

def Feature_Extractor(dim=64, patch_size=4, depth=2):
    return nn.Sequential(
        nn.Conv2d(3, dim//2, 3, padding=1),
        nn.MaxPool2d(2),
        my_norm(1764),
        nn.GELU(),
        nn.Conv2d(dim//2, dim, 3, padding=1),
        nn.MaxPool2d(2),
        my_norm(441),
        nn.GELU(),
        *[MultiScaleExtractor(dim=dim) for _ in range(depth)]
    )


def Conv64F_FeatureExtractor(input_size=84, dim=64, depth=2):
    return nn.Sequential(

        nn.Conv2d(3, dim, kernel_size=3, padding=1),  # [B,64,H,W]
        nn.MaxPool2d(kernel_size=2, stride=2),
        my_norm(1764),
        nn.GELU(),


        nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        my_norm(441),
        nn.GELU(),


        *[nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        ) for _ in range(depth)]
    )
def Resnet18_FeatureExtractor(dim=64, patch_size=4, depth=2):
    resnet18 = models.resnet18(pretrained=False)
    resnet_layers = list(resnet18.children())[:-2]


    resnet_layers[0] = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)


    for layer in [resnet_layers[4], resnet_layers[5], resnet_layers[6]]:
        for block in layer:
            if hasattr(block, 'conv1'):
                block.conv1.stride = (1, 1)
            if block.downsample is not None:

                block.downsample = nn.Sequential(
                    nn.Conv2d(block.downsample[0].in_channels, block.downsample[0].out_channels,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(block.downsample[0].out_channels)
                )

    resnet18 = nn.Sequential(*resnet_layers)

    return nn.Sequential(
        resnet18,
        nn.Conv2d(512, dim, kernel_size=1),
        nn.BatchNorm2d(dim),
        nn.GELU()
    )

#-------------------------------------Mahalanobis block------------------------------------------------------#
class MahalanobisBlock(nn.Module):
    def __init__(self):
        super(MahalanobisBlock, self).__init__()

    def cal_covariance(self, input):
        CovaMatrix_list = []
        for i in range(len(input)):
            support_set_sam = input[i]
            B, C, h, w = support_set_sam.size()
            local_feature_list = []

            for local_feature in support_set_sam:
                local_feature_np = local_feature.detach().cpu().numpy() 
                transposed_tensor = np.transpose(local_feature_np, (1, 2, 0))
                reshaped_tensor = np.reshape(transposed_tensor, (h * w, C))

                for line in reshaped_tensor:
                    local_feature_list.append(line)

            local_feature_np = np.array(local_feature_list)
            mean = np.mean(local_feature_np, axis=0)
            local_feature_list = [x - mean for x in local_feature_list]

            covariance_matrix = np.cov(local_feature_np, rowvar=False)
            covariance_matrix = torch.from_numpy(covariance_matrix)
            CovaMatrix_list.append(covariance_matrix)

        return CovaMatrix_list



    def mahalanobis_similarity(self, input, CovaMatrix_list, regularization=1e-6):
        B, C, h, w = input.size()
        mahalanobis = []

        for i in range(B):
            query_sam = input[i]
            query_sam = query_sam.view(C, -1)
            query_sam_norm = torch.norm(query_sam, 2, 1, True)
            query_sam = query_sam / query_sam_norm
            mea_sim = torch.zeros(1, len(CovaMatrix_list) * h * w).cuda()
            # mea_sim = torch.zeros(1, len(CovaMatrix_list) * h * w)
            # print("CovaMatrix_list:",len(CovaMatrix_list))
            for j in range(len(CovaMatrix_list)):

                covariance_matrix = CovaMatrix_list[j].float().cuda() + regularization * torch.eye(C).cuda()
                # covariance_matrix = CovaMatrix_list[j] + regularization * torch.eye(C)
                inv_covariance_matrix = torch.linalg.inv(covariance_matrix)
                diff = query_sam - torch.mean(query_sam, dim=1, keepdim=True)
                temp_dis = torch.matmul(torch.matmul(diff.T, inv_covariance_matrix), diff)
                mea_sim[0, j * h * w:(j + 1) * h * w] = temp_dis.diag()
                # print("mea_sim", mea_sim.shape)
            mahalanobis.append(mea_sim.view(1, -1))

        mahalanobis = torch.cat(mahalanobis, 0)

        return mahalanobis


    def forward(self, x1, x2):

        CovaMatrix_list = self.cal_covariance(x2)
        maha_sim = self.mahalanobis_similarity(x1, CovaMatrix_list)

        return maha_sim


class EuclideanBlock(nn.Module):
    def __init__(self):
        super(EuclideanBlock, self).__init__()

    def euclidean_similarity(self, x1, x2):
        B, C, h, w = x1.size()
        num_support = len(x2)
        euclidean_distances = []

        for i in range(B):
            query_sam = x1[i].view(C, -1)
            dist_vector = torch.zeros(1, num_support * h * w).cuda()

            for j in range(num_support):
                support_sam = x2[i][j].view(C, -1)


                if support_sam.shape[1] != query_sam.shape[1]:
                    min_dim = min(support_sam.shape[1], query_sam.shape[1])
                    support_sam = support_sam[:, :min_dim]
                    query_sam = query_sam[:, :min_dim]


                temp_dis = torch.norm(query_sam - support_sam, dim=0)
                dist_vector[0, j * h * w:(j + 1) * h * w] = temp_dis

            euclidean_distances.append(dist_vector.view(1, -1))

        euclidean_distances = torch.cat(euclidean_distances, 0)
        return euclidean_distances

    def forward(self, x1, x2):
        euclidean_sim = self.euclidean_similarity(x1, x2)
        return euclidean_sim


class CosineBlock(nn.Module):
    def __init__(self):
        super(CosineBlock, self).__init__()

    def cosine_similarity(self, x1, x2):
        B, C, h, w = x1.size()
        num_support = len(x2)
        cosine_distances = []

        for i in range(B):
            query_sam = x1[i].view(C, -1)
            query_sam = F.normalize(query_sam, p=2, dim=0)
            dist_vector = torch.zeros(1, num_support * h * w).cuda()

            for j in range(num_support):
                support_sam = x2[i][j].view(C, -1)
                support_sam = F.normalize(support_sam, p=2, dim=0)


                cosine_sim = torch.sum(query_sam * support_sam, dim=0)
                cosine_dist = 1 - cosine_sim

                dist_vector[0, j * h * w:(j + 1) * h * w] = cosine_dist

            cosine_distances.append(dist_vector.view(1, -1))

        cosine_distances = torch.cat(cosine_distances, 0)
        return cosine_distances

    def forward(self, x1, x2):
        cosine_sim = self.cosine_similarity(x1, x2)
        return cosine_sim
#---------------- Transformer---------------------------------------
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim = 64):
        super(ScaledDotProductAttention, self).__init__()
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.dim = dim
        self.q = 0
    def forward(self, q, k, v):

        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        scaled_dot_product = torch.matmul(q.unsqueeze(2), k.unsqueeze(1)) / torch.sqrt(torch.tensor(self.dim, dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(scaled_dot_product, dim=-1)
        self.q = attention_weights

        output = torch.matmul(attention_weights, v.unsqueeze(2))
        output = output.squeeze(2)
        return output


class LowRankScaledDotProductAttention(nn.Module):
    def __init__(self, dim=64, rank=32):
        super(LowRankScaledDotProductAttention, self).__init__()
        self.dim = dim
        self.rank = rank


        self.q_project = nn.Sequential(
            nn.Linear(dim, rank),
            nn.Linear(rank, dim)
        )
        self.k_project = nn.Sequential(
            nn.Linear(dim, rank),
            nn.Linear(rank, dim)
        )
        self.v_project = nn.Sequential(
            nn.Linear(dim, rank),
            nn.Linear(rank, dim)
        )


        self._initialize_weights()

        self.q = 0

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, q, k, v):


        q = self.q_project(q)
        k = self.k_project(k)
        v = self.v_project(v)


        scaled_dot_product = torch.matmul(q.unsqueeze(2), k.unsqueeze(1)) / torch.sqrt(
            torch.tensor(self.dim, dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(scaled_dot_product, dim=-1)
        self.q = attention_weights
        output = torch.matmul(attention_weights, v.unsqueeze(2)).squeeze(2)

        return output


class GatedScaledDotProductAttention(nn.Module):
    def __init__(self, dim=64):
        super(GatedScaledDotProductAttention, self).__init__()
        self.dim = dim


        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)


        self.gate_linear = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.Sigmoid()
        )

        self.q = 0

    def forward(self, q, k, v):

        q_proj = self.q_linear(q)
        k_proj = self.k_linear(k)
        v_proj = self.v_linear(v)


        scaled_dot_product = torch.matmul(q_proj.unsqueeze(2),
                                          k_proj.unsqueeze(1)) \
                             / torch.sqrt(torch.tensor(self.dim, dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(scaled_dot_product, dim=-1)
        self.q = attention_weights


        base_output = torch.matmul(attention_weights, v_proj.unsqueeze(2)).squeeze(2)


        gate_input = torch.cat([q, k, v], dim=-1)

        gate = self.gate_linear(gate_input)


        gated_output = gate * base_output + (1 - gate) * v


        return gated_output


class CrossAttention(nn.Module):
    def __init__(self, dim=64):
        super(CrossAttention, self).__init__()
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.dim = dim
    def forward(self, q, k, v):

        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        scaled_dot_product = torch.matmul(q.unsqueeze(2), k.unsqueeze(1))
        attention_weights = torch.nn.functional.softmax(scaled_dot_product, dim=-1)
        # output = torch.matmul(attention_weights, v)

        return attention_weights


class LowRankCrossAttention(nn.Module):
    def __init__(self, dim=64, rank=32):
        super(LowRankCrossAttention, self).__init__()
        self.dim = dim
        self.rank = rank


        self.q_linear = nn.Sequential(
            nn.Linear(dim, rank),
            nn.Linear(rank, dim)
        )
        self.k_linear = nn.Sequential(
            nn.Linear(dim, rank),
            nn.Linear(rank, dim)
        )
        self.v_linear = nn.Sequential(
            nn.Linear(dim, rank),
            nn.Linear(rank, dim)
        )

    def forward(self, q, k, v):


        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)


        scaled_dot_product = torch.matmul(
            q.unsqueeze(2),
            k.unsqueeze(1)
        )

        attention_weights = torch.nn.functional.softmax(
            scaled_dot_product,
            dim=-1
        )

        return attention_weights


class GatedCrossAttention(nn.Module):
    def __init__(self, dim=64):
        super(GatedCrossAttention, self).__init__()
        self.dim = dim


        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)


        self.gate_generator = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, q, k, v):


        q_proj = self.q_linear(q)  # [B,D]
        k_proj = self.k_linear(k)  # [B,D]


        scaled_dot_product = torch.matmul(
            q_proj.unsqueeze(2),
            k_proj.unsqueeze(1)
        )


        q_exp = q_proj.unsqueeze(2).expand(-1, -1, self.dim)
        k_exp = k_proj.unsqueeze(1).expand(-1, self.dim, -1)
        pair_features = torch.cat([q_exp, k_exp], dim=-1)


        gate = self.gate_generator(pair_features)
        gate = torch.sigmoid(gate)


        gated_attention = scaled_dot_product * gate
        attention_weights = torch.softmax(gated_attention, dim=-1)

        return attention_weights


class Encoder(nn.Module):
    def __init__(self, dim):
        super(Encoder, self).__init__()
        self.dim = dim
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.ScaledDotProductAttention = ScaledDotProductAttention()

        self.norm = nn.LayerNorm(normalized_shape=self.dim)
        self.FFN = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
            nn.GELU()
        )

    def forward(self, Support):
      encoded = []
      for index in range(len(Support)):
          s = Support[index]                                  # [B, C, H, W]
          s = self.GAP(s)                                     # [B, C, 1, 1]
          s = s.view(s.size(0), s.size(1))                    # [B, C]
          s = self.ScaledDotProductAttention(s, s, s) + s     # [B, C]
          s = self.norm(s)                                    # [B, C]
          s = self.FFN(s) + s                                 # [B, C]
          s = self.norm(s)                                    # [B, C]
          s = torch.mean(s, dim=0, keepdim=True)
          encoded.append(s)

      return encoded                                          # [Num_class x (B, C)]

class Encoder_Decoder(nn.Module):
    def __init__(self, dim):
        super(Encoder_Decoder, self).__init__()
        self.dim = dim
        self.encoder_out = Encoder(self.dim)
        # self.attention = CrossAttention()
        self.attention = GatedCrossAttention()
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        # self.ScaledDotProductAttention = LowRankScaledDotProductAttention
        # self.ScaledDotProductAttention =GatedScaledDotProductAttention
        self.norm = nn.LayerNorm(normalized_shape=self.dim)
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.Linear = nn.Linear(self.dim ** 2 , 1)
        self.a = 0
        self.a_q=0
    def forward(self, q, S):
        q = self.GAP(q)                                       # [B, C, 1, 1]
        # print("GAP(Q):",q.shape)
        q = q.view(q.size(0), q.size(1))                      # [B, C]
        q_first = q                                           # [B, C]
        q = self.ScaledDotProductAttention(q, q, q) + q_first # [B, C]
        q = self.norm(q)                                      # [B, C]
        self.a_q = self.ScaledDotProductAttention.q
        output = []
        self.out_att = []
        encoder_outs = self.encoder_out(S)                    # [Num_class x (B, C)]

        for encoder_out in encoder_outs:
            out = self.attention(q, encoder_out, encoder_out) # [B, C, C]
            self.a = out

            out = out.view(out.size(0), -1)                   # [B, C*C]

            out = self.Linear(out)
            # print("out:",out.shape)
            output.append(out)
            self.out_att.append(self.a)
        # print("a:", np.array(self.out_att).shape)


        return output                                         # [Num_class x (B, 1)]

    
#-------------------------------Proposed Ensemble------------------------------------------------
class Ensemble_Net(nn.Module):
    def __init__(self, h=21,w =21,norm_layer=nn.BatchNorm2d, dim=64, alpha1=0.5, alpha2 =0.5):
        self.h = h
        self.w = w
        super(Ensemble_Net, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.covariance = MahalanobisBlock()
        # self.covariance = EuclideanBlock()#****
        self.Encoder_Decoder = Encoder_Decoder(dim)
        self.features = Feature_Extractor()
        self.features1 = Conv64F_FeatureExtractor()
        self.features2 = Resnet18_FeatureExtractor()
        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Conv1d(1, 1, kernel_size=self.h*self.w, stride=self.h*self.w, bias=use_bias),
        )
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.ma = 0
        self.attention = 0
    def forward(self, input1, input2):
        q = self.features(input1)


        S = []
        for i in range(len(input2)):
            features = self.features(input2[i])
            S.append(features)

        # Lower branch
        m_l = self.covariance(q, S)
        self.ma = m_l

        m_l = self.classifier(m_l.view(m_l.size(0), 1, -1))

        m_l = m_l.squeeze(1)

        # Upper branch
        m_u = self.Encoder_Decoder(q, S)
        self.attention = self.Encoder_Decoder.out_att
        m_u = torch.cat(m_u, 1)

        output = self.alpha1*m_l + self.alpha2*m_u

        return m_l, m_u, output
