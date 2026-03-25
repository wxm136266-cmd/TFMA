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
        nn.Conv2d(3, dim//2, 3, padding=1), #这里的channel可能有疑问
        nn.MaxPool2d(2),
        my_norm(1764), #128:4096 84：1764
        nn.GELU(),
        nn.Conv2d(dim//2, dim, 3, padding=1),
        nn.MaxPool2d(2),
        my_norm(441), #128:1024 84：441
        nn.GELU(),
        *[MultiScaleExtractor(dim=dim) for _ in range(depth)]
    )


def Conv64F_FeatureExtractor(input_size=84, dim=64, depth=2):
    return nn.Sequential(
        # Stage 1: 下采样1/2（84 → 42）
        nn.Conv2d(3, dim, kernel_size=3, padding=1),  # [B,64,H,W]
        nn.MaxPool2d(kernel_size=2, stride=2),
        my_norm(1764),  # 保持原始参数（42x42的数字运算）
        nn.GELU(),

        # Stage 2: 下采样1/4（42 → 21）
        nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        my_norm(441),  # 21x21的特征图面积
        nn.GELU(),

        # Conv-64F特征处理块
        *[nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=1),  # 标准卷积（无通道变换）
            nn.BatchNorm2d(dim),  # (可选) 更强的正则化
            nn.GELU()
        ) for _ in range(depth)]
    )
def Resnet18_FeatureExtractor(dim=64, patch_size=4, depth=2):
    resnet18 = models.resnet18(pretrained=False)
    resnet_layers = list(resnet18.children())[:-2]  # 移除全连接层和全局池化层

    # 修改第一层卷积，使 stride=1，防止过早下采样
    resnet_layers[0] = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)

    # 修改 layer2, layer3, layer4 的 stride，并确保 downsample 适配尺寸
    for layer in [resnet_layers[4], resnet_layers[5], resnet_layers[6]]:  # layer2, layer3, layer4
        for block in layer:
            if hasattr(block, 'conv1'):
                block.conv1.stride = (1, 1)  # 保持尺寸
            if block.downsample is not None:
                # 确保 downsample 也能匹配尺寸
                block.downsample = nn.Sequential(
                    nn.Conv2d(block.downsample[0].in_channels, block.downsample[0].out_channels,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(block.downsample[0].out_channels)
                )

    resnet18 = nn.Sequential(*resnet_layers)

    return nn.Sequential(
        resnet18,
        nn.Conv2d(512, dim, kernel_size=1),  # 1x1 卷积降维到 dim (64)
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
        B, C, h, w = x1.size()  # Query: (B, C, H, W)
        num_support = len(x2)  # Support: (B, num_support, C, H, W)
        euclidean_distances = []

        for i in range(B):
            query_sam = x1[i].view(C, -1)  # (C, H*W)
            dist_vector = torch.zeros(1, num_support * h * w).cuda()

            for j in range(num_support):
                support_sam = x2[i][j].view(C, -1)  # **注意这里，取 batch 内的第 j 个 support 样本**

                # **确保维度一致**
                if support_sam.shape[1] != query_sam.shape[1]:
                    min_dim = min(support_sam.shape[1], query_sam.shape[1])
                    support_sam = support_sam[:, :min_dim]
                    query_sam = query_sam[:, :min_dim]

                # **计算欧式距离**
                temp_dis = torch.norm(query_sam - support_sam, dim=0)
                dist_vector[0, j * h * w:(j + 1) * h * w] = temp_dis

            euclidean_distances.append(dist_vector.view(1, -1))

        euclidean_distances = torch.cat(euclidean_distances, 0)  # (B, num_support * H * W)
        return euclidean_distances

    def forward(self, x1, x2):
        euclidean_sim = self.euclidean_similarity(x1, x2)
        return euclidean_sim


class CosineBlock(nn.Module):
    def __init__(self):
        super(CosineBlock, self).__init__()

    def cosine_similarity(self, x1, x2):
        B, C, h, w = x1.size()  # Query: (B, C, H, W)
        num_support = len(x2) # Support: (B, num_support, C, H, W)
        cosine_distances = []

        for i in range(B):
            query_sam = x1[i].view(C, -1)  # (C, H*W)
            query_sam = F.normalize(query_sam, p=2, dim=0)  # L2 归一化
            dist_vector = torch.zeros(1, num_support * h * w).cuda()

            for j in range(num_support):
                support_sam = x2[i][j].view(C, -1)
                support_sam = F.normalize(support_sam, p=2, dim=0)  # L2 归一化

                # **计算余弦相似度**
                cosine_sim = torch.sum(query_sam * support_sam, dim=0)  # 计算点积
                cosine_dist = 1 - cosine_sim  # 余弦距离

                dist_vector[0, j * h * w:(j + 1) * h * w] = cosine_dist

            cosine_distances.append(dist_vector.view(1, -1))

        cosine_distances = torch.cat(cosine_distances, 0)  # (B, num_support * H * W)
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
        """
        Args:
            q (Tensor): Query tensor of shape (batch_size, dim).
            k (Tensor): Key tensor of shape (batch_size, dim).
            v (Tensor): Value tensor of shape (batch_size, dim).

        Returns:
            output (Tensor): Scaled Dot-Product Attention output tensor of shape (batch_size, dim).
        """
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        scaled_dot_product = torch.matmul(q.unsqueeze(2), k.unsqueeze(1)) / torch.sqrt(torch.tensor(self.dim, dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(scaled_dot_product, dim=-1)
        self.q = attention_weights
        # print("q:", self.q)
        output = torch.matmul(attention_weights, v.unsqueeze(2))
        output = output.squeeze(2)
        return output


class LowRankScaledDotProductAttention(nn.Module):
    def __init__(self, dim=64, rank=32):
        super(LowRankScaledDotProductAttention, self).__init__()
        self.dim = dim
        self.rank = rank

        # Low-rank projections (分解每个线性变换为两个低秩矩阵)
        self.q_project = nn.Sequential(
            nn.Linear(dim, rank),  # 降维到低秩空间
            nn.Linear(rank, dim)  # 恢复原始维度
        )
        self.k_project = nn.Sequential(
            nn.Linear(dim, rank),
            nn.Linear(rank, dim)
        )
        self.v_project = nn.Sequential(
            nn.Linear(dim, rank),
            nn.Linear(rank, dim)
        )

        # 可选：与原模块相同的参数初始化方式
        self._initialize_weights()

        self.q = 0  # 保持与原始类相同的监控变量

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, q, k, v):
        """低秩注意力前向计算（维度保持不变）"""
        # 低秩投影
        q = self.q_project(q)  # [B, D] → [B, R] → [B, D]
        k = self.k_project(k)
        v = self.v_project(v)

        # 保持原有注意力计算逻辑
        scaled_dot_product = torch.matmul(q.unsqueeze(2), k.unsqueeze(1)) / torch.sqrt(
            torch.tensor(self.dim, dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(scaled_dot_product, dim=-1)
        self.q = attention_weights  # 保持对权重的监控
        output = torch.matmul(attention_weights, v.unsqueeze(2)).squeeze(2)

        return output  # 输出维度保持为[B, D]


class GatedScaledDotProductAttention(nn.Module):
    def __init__(self, dim=64):
        super(GatedScaledDotProductAttention, self).__init__()
        self.dim = dim

        # 原始QKV线性变换
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)

        # 新增门控参数
        self.gate_linear = nn.Sequential(
            nn.Linear(3 * dim, dim),  # 融合Q/K/V特征
            nn.Sigmoid()  # 门控激活函数
        )

        self.q = 0  # 保持监控变量

    def forward(self, q, k, v):
        # 原始注意力计算
        q_proj = self.q_linear(q)
        k_proj = self.k_linear(k)
        v_proj = self.v_linear(v)

        # 计算注意力权重
        scaled_dot_product = torch.matmul(q_proj.unsqueeze(2),
                                          k_proj.unsqueeze(1)) \
                             / torch.sqrt(torch.tensor(self.dim, dtype=torch.float32))
        attention_weights = torch.nn.functional.softmax(scaled_dot_product, dim=-1)
        self.q = attention_weights

        # 基础注意力输出
        base_output = torch.matmul(attention_weights, v_proj.unsqueeze(2)).squeeze(2)

        # 门控机制
        # 融合Q/K/V特征生成门控信号 (B, 3D)
        gate_input = torch.cat([q, k, v], dim=-1)
        # 生成门控权重 (B, D)
        gate = self.gate_linear(gate_input)

        # 门控增强输出：通过门控控制基础注意力输出的信息流
        gated_output = gate * base_output + (1 - gate) * v  # 残差连接
        # gated_output = gate * base_output                # 简单版本

        return gated_output


class CrossAttention(nn.Module):
    def __init__(self, dim=64):
        super(CrossAttention, self).__init__()
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.dim = dim
    def forward(self, q, k, v):
        """
        Args:
            q (Tensor): Query tensor of shape (batch_size,  dim).
            k (Tensor): Key tensor of shape (batch_size, dim).
            v (Tensor): Value tensor of shape (batch_size, dim).

        Returns:
            output (Tensor): Scaled Dot-Product Attention output tensor of shape (batch_size, dim).
        """
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

        # 将线性层分解为低秩矩阵序列
        self.q_linear = nn.Sequential(
            nn.Linear(dim, rank),  # 降维到低秩空间
            nn.Linear(rank, dim)  # 恢复原始维度
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
        """
        低秩交叉注意力前向传播，输出维度与原始模块一致

        Args:
            q (Tensor): [B, D]
            k (Tensor): [B, D]
            v (Tensor): [B, D]

        Returns:
            attention_weights (Tensor): [B, D, D]
        """
        # 低秩投影
        q = self.q_linear(q)  # [B, D] → [B, R] → [B, D]
        k = self.k_linear(k)
        v = self.v_linear(v)  # 保持v的低秩投影（若未使用也可删除）

        # 原有注意力计算流程保持激活
        scaled_dot_product = torch.matmul(
            q.unsqueeze(2),  # [B, D, 1]
            k.unsqueeze(1)  # [B, 1, D]
        )  # => [B, D, D]

        attention_weights = torch.nn.functional.softmax(
            scaled_dot_product,
            dim=-1
        )

        return attention_weights  # 输出维度仍然为[B, D, D]


class GatedCrossAttention(nn.Module):
    def __init__(self, dim=64):
        super(GatedCrossAttention, self).__init__()
        self.dim = dim

        # 原始Q/K投影层（V投影不再使用）
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)

        # 门控生成器网络
        self.gate_generator = nn.Sequential(
            nn.Linear(dim * 2, dim),  # 融合Q/K特征
            nn.Sigmoid()  # 输出范围[0,1]
        )

    def forward(self, q, k, v):
        """
        门控注意力流程说明：
        1. 计算原始注意力矩阵 [B,D,D]
        2. 动态生成门控矩阵 [B,D,D]
        3. 加权后的注意力仍然保持 [B,D,D]
        """
        ###### 基础投影 ######
        q_proj = self.q_linear(q)  # [B,D]
        k_proj = self.k_linear(k)  # [B,D]

        ###### 原注意力权重 ######
        scaled_dot_product = torch.matmul(
            q_proj.unsqueeze(2),  # [B,D,1]
            k_proj.unsqueeze(1)  # [B,1,D]
        )  # → [B,D,D]

        ###### 动态门控矩阵生成 ######
        # 构造位置感知特征对 (q_i ⊕ k_j)
        q_exp = q_proj.unsqueeze(2).expand(-1, -1, self.dim)  # [B,D,D]
        k_exp = k_proj.unsqueeze(1).expand(-1, self.dim, -1)  # [B,D,D]
        pair_features = torch.cat([q_exp, k_exp], dim=-1)  # [B,D,D,2D]

        # 生成门控矩阵
        gate = self.gate_generator(pair_features)  # [B,D,D,D] → [B,D,D]（全连接将2D压缩到D）
        gate = torch.sigmoid(gate)  # 确保数值稳定性

        ###### 门控加权后的注意力 ######
        gated_attention = scaled_dot_product * gate  # 元素级相乘
        attention_weights = torch.softmax(gated_attention, dim=-1)

        return attention_weights  # 输出保持[B,D,D]


class Encoder(nn.Module):
    def __init__(self, dim):
        super(Encoder, self).__init__()
        self.dim = dim
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        # self.ScaledDotProductAttention = LowRankScaledDotProductAttention() #****
        # self.ScaledDotProductAttention = GatedScaledDotProductAttention()
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
        self.attention = GatedCrossAttention()#****
        self.ScaledDotProductAttention = ScaledDotProductAttention()
        # self.ScaledDotProductAttention = LowRankScaledDotProductAttention()#****
        # self.ScaledDotProductAttention =GatedScaledDotProductAttention()#*****
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
        self.h = h #!!!!特征提取后图像的大小 要记得修改
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

        # print("feature(q):",q.shape)
        S = []
        for i in range(len(input2)):
            features = self.features(input2[i])
            S.append(features)
        # print("s:",len(S))
        # Lower branch
        m_l = self.covariance(q, S)
        self.ma = m_l
        # print("Ma:",self.ma.shape)
        m_l = self.classifier(m_l.view(m_l.size(0), 1, -1))
        # print('m_l:',m_l.shape)
        m_l = m_l.squeeze(1)

        # Upper branch
        m_u = self.Encoder_Decoder(q, S)
        self.attention = self.Encoder_Decoder.out_att
        m_u = torch.cat(m_u, 1)
        # print('m_u:', m_u.shape)
        output = self.alpha1*m_l + self.alpha2*m_u

        return m_l, m_u, output
