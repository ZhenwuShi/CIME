import torch
import torch.nn as nn
from torch import  nn
from src.model.utils.timestep_embed import TimestepEmbedding, Timesteps, TimestepEmbedderMDM
from src.model.utils.positional_encoding import PositionalEncoding
from src.model.utils.transf_utils import SkipTransformerEncoder, TransformerEncoderLayer
from src.model.utils.all_positional_encodings import build_position_encoding
from src.data.tools.tensors import lengths_to_mask
from src.model.utils.timestep_embed import TimestepEmbedderMDM
from src.model.DiT_models import DiTMotion
import math

class ContinuousImplicitAligner(nn.Module):
    """
    ===================================================================================
    [TPAMI 核心贡献模块]: Fused Gromov-Wasserstein Riemannian Aligner (FGW-R-CIMR)
    
    【核心哲学：时间的“变”与“不变”】
    - 变 (Change): Target 长度发生改变，一阶语义特征需要自适应拉伸匹配。
    - 不变 (Invariance): Source 内部的物理爆发力规律（通过黎曼时间测量）、
                       以及全局的时序拓扑结构（通过 FGW 二阶节奏网测量），绝对不变！
    ===================================================================================
    """
    def __init__(self, feature_dim=512, num_heads=8, fgw_alpha=0.5, fgw_iters=2, epsilon=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # --- FGW 超参数 ---
        self.fgw_alpha = fgw_alpha       # 融合系数：0(纯看局部特征相似度) <--> 1(纯看全局节奏一致性)
        self.fgw_iters = fgw_iters       # 二阶 GW 外部对齐循环次数
        self.epsilon = epsilon           # 熵正则化温度：值越小，Attention 越锐利；值越大，越平滑
        self.sinkhorn_iters = 3          # 一阶 Sinkhorn 内部行列归一化次数
        
        # --- 隐式时间映射网络 ---
        # 负责将低维时间 t 映射为与动作特征同维度的高频表征
        self.time_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # --- 跨时空寻路投影层 ---
        # 抛弃了原生 nn.MultiheadAttention，完全由我们手写的 FGW 来接管寻路权重的计算
        self.q_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.v_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def fourier_encode(self, t, device):
        """
        [基础设施]: 高频傅里叶位置编码 (Implicit Neural Representation)
        作用：将 [0, 1] 之间的连续进度值，展开成高频正余弦波，防止微小时间差被网络忽略。
        """
        half_dim = self.feature_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0) 
        time_emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return time_emb

    def forward(self, source_motion, T_tgt):
        B, T_src, D = source_motion.shape
        device = source_motion.device
        
        if isinstance(T_tgt, torch.Tensor):
            T_tgt = int(T_tgt.item())
            
        """
        ===================================================================================
        【模块一：非对称时间流形提取 (Asymmetric Temporal Manifolds)】
        打破“匀速时间”的常识，赋予动作真正的物理刻度。
        ===================================================================================
        """
        
        # ---------------------------------------------------------
        # 1. Source 端：黎曼测地线时间 (Riemannian Geodesic Time)
        # 大白话：动作越剧烈，这里算出来的时间流逝就越快（分配的进度条空间就越大）
        # ---------------------------------------------------------
        # 1.a 计算相邻帧特征差异的 L2 Norm，作为瞬时物理动量 (Velocity)
        velocity = torch.norm(source_motion[:, 1:] - source_motion[:, :-1], dim=-1) # (B, T_src-1)
        
        # 1.b 补齐第0帧，并强制加入 1e-4 的底噪。
        # 原因：如果不加，当人物完全静止(velocity=0)时，积分曲线会平着走，导致多个帧时间戳完全重叠(奇异值)。
        base_vel = 1e-4
        velocity = torch.cat([torch.zeros(B, 1, device=device), velocity], dim=1) + base_vel
        
        # 1.c 积分求物理做功累积量 (也就是黎曼流形上的弧长)
        arc_length = torch.cumsum(velocity, dim=1) # (B, T_src)
        
        # 1.d 归一化，得到源动作的物理节奏进度条 t_src ∈ [0, 1]
        t_src = arc_length / arc_length[:, -1:] # (B, T_src)
        time_src_feat = self.time_mlp(self.fourier_encode(t_src, device))
        
        # ---------------------------------------------------------
        # 2. Target 端：均匀欧几里得时间 (Euclidean Time)
        # 大白话：Target 是被扩散噪声污染的，没有物理结构，所以老老实实当匀速的“查询游标”
        # ---------------------------------------------------------
        t_tgt = torch.linspace(0, 1, T_tgt, device=device).unsqueeze(0).expand(B, -1) # (B, T_tgt)
        time_tgt_feat = self.time_mlp(self.fourier_encode(t_tgt, device))
        
        
        """
        ===================================================================================
        【模块二：FGW 代价矩阵构建 (Cost Matrix Construction)】
        准备好“一阶语义距离”和“二阶节奏距离”，为最优传输提供裁决依据。
        ===================================================================================
        """
        # ---------------------------------------------------------
        # 3. 构造一阶代价值 (First-Order Semantic Cost: M_cost)
        # 大白话：Target 的当前时间点，跟 Source 哪一帧的“特征+时间”最匹配？
        # ---------------------------------------------------------
        Q = self.q_proj(time_tgt_feat).view(B, T_tgt, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(source_motion + time_src_feat).view(B, T_src, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(source_motion + time_src_feat).view(B, T_src, self.num_heads, self.head_dim).transpose(1, 2)

        # 相似度 (Attention Logits) 本质上是负代价，取负号变成 Cost
        sim_matrix = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        M_cost = -sim_matrix  # (B, H, T_tgt, T_src)

        # ---------------------------------------------------------
        # 4. 构造二阶拓扑度量张量 (Intra-Space Metric Tensors: C_tgt, C_src)
        # 大白话：不看跨序列匹不匹配，只算自己内部任意两帧相隔多远。这就叫“节奏网”！
        # ---------------------------------------------------------
        C_tgt = (t_tgt.unsqueeze(-1) - t_tgt.unsqueeze(-2)) ** 2  # (B, 1, T_tgt, T_tgt)
        C_src = (t_src.unsqueeze(-1) - t_src.unsqueeze(-2)) ** 2  # (B, 1, T_src, T_src)
        C_tgt = C_tgt.unsqueeze(1).to(device)
        C_src = C_src.unsqueeze(1).to(device)

        # 预计算 Gromov-Wasserstein 展开式中，不需要随迭代变化的常量惩罚项
        ones_tgt_src = torch.ones(B, 1, T_tgt, T_src, device=device)
        bias_GW = torch.matmul(C_tgt ** 2, ones_tgt_src) / T_src + \
                  torch.matmul(ones_tgt_src, C_src ** 2) / T_tgt


        """
        ===================================================================================
        【模块三：对数域 FGW 迭代求解 (Log-Domain FGW Iteration)】
        数学强制校验：满足语义改变的同时，绝对锁死全局节奏，并保证物理能量守恒！
        ===================================================================================
        """
        # 初始化均匀的传输计划 P (即最初始的 Attention 权重)
        P = torch.ones(B, self.num_heads, T_tgt, T_src, device=device) / (T_tgt * T_src)
        
        for _ in range(self.fgw_iters):
            # ---------------------------------------------------------
            # 5.1 计算动态二阶代价 (GW_cost)
            # 大白话：校验当前 P 矩阵对齐后，Target 借过来的节奏，和 Source 原始的物理节奏差多少？
            # ---------------------------------------------------------
            GW_cost = bias_GW - 2 * torch.matmul(C_tgt, torch.matmul(P, C_src))
            
            # ---------------------------------------------------------
            # 5.2 全局代价融合
            # 大白话：(1-α) 听文本指令去“变” + α 遵守客观物理规律“不变”
            # ---------------------------------------------------------
            Total_Cost = (1 - self.fgw_alpha) * M_cost + self.fgw_alpha * GW_cost
            
            # ---------------------------------------------------------
            # 5.3 🌶️ 核心护城河：转入 Log 空间 (防止梯度爆炸/消失)
            # 大白话：把指数运算 e^(-Cost) 变成对数域的加减法，彻底杀死 NaN
            # ---------------------------------------------------------
            Total_Cost = torch.clamp(Total_Cost, -1e4, 1e4) # 暴力掐断初始极端异常值
            log_P = -Total_Cost / self.epsilon
            
            # ---------------------------------------------------------
            # 5.4 Sinkhorn-Knopp 双向归一化 (运动学能量守恒)
            # 大白话：不但要求 Target 每帧拿饱100%，还强迫 Source 每帧被抽走100% (一帧短促动作都不许漏掉！)
            # ---------------------------------------------------------
            for __ in range(self.sinkhorn_iters):
                # Target 需求端归一化 (行归一)
                log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
                # Source 供给端归一化 (列归一)
                log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
                
            # 退出 Log 空间，安全还原为正数概率权重 P
            P = torch.exp(log_P)

        """
        ===================================================================================
        【模块四：特征流形重采样 (Manifold Resampling)】
        ===================================================================================
        """
        # 用历经千锤百炼的传输矩阵 P（它现在是最完美的 Attention 权重），去提取 V
        aligned_source = torch.matmul(P, V)
        
        # 还原张量形状
        aligned_source = aligned_source.transpose(1, 2).reshape(B, T_tgt, D)
        aligned_source = self.out_proj(aligned_source)
        
        return aligned_source


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )

# architecure ablation
class DiT_Denoiser_CLS_Arch(nn.Module):

    def __init__(self,
                 nfeats: int = 263,
                 condition: str = "text",
                 motion_condition: str = None,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 text_encoded_dim: int = 768,
                 pred_delta_motion: bool = False,
                 use_sep: bool = True,
                 **kwargs) -> None:

        super().__init__()
        self.latent_dim = latent_dim
        self.text_encoded_dim = text_encoded_dim
        self.condition = condition
        self.feat_comb_coeff = nn.Parameter(torch.tensor([1.0]))
        self.pose_proj_in_source = nn.Linear(nfeats, self.latent_dim)
        self.pose_proj_in_target = nn.Linear(nfeats, self.latent_dim)
        self.pose_proj_out = nn.Linear(self.latent_dim, nfeats)
        self.motion_condition = motion_condition
        self.inter_proj_1 = nn.Linear(self.latent_dim, nfeats)
        self.inter_proj_2 = nn.Linear(self.latent_dim, nfeats)
        self.inter_proj_3 = nn.Linear(self.latent_dim, nfeats)

        # emb proj
        if self.condition in ["text", "text_uncond"]:
            # text condition
            # project time+text to latent_dim
            if text_encoded_dim != self.latent_dim:
                # todo 10.24 debug why relu
                self.emb_proj = nn.Linear(text_encoded_dim, self.latent_dim)
        else:
            raise TypeError(f"condition type {self.condition} not supported")
        self.use_sep = True
        self.query_pos = PositionalEncoding(self.latent_dim, dropout = 0)
        self.cond_pos = PositionalEncoding(self.latent_dim, dropout = 0)# don't want to introduce noise here
        if self.motion_condition == "source":
            self.sep_token = nn.Parameter(torch.randn(1, self.latent_dim))

        # dit encoder
        self.dit_encoder = DiTMotion(
            in_channels=self.latent_dim,
            hidden_size=self.latent_dim,
            depth=num_layers,
            num_heads=num_heads,
            mlp_ratio=ff_size / self.latent_dim,
        )
        # use torch transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation)
        self.cond_encoder = nn.TransformerEncoder(encoder_layer,
                                                num_layers=kwargs.get('encoder_layers', 4))
        self.has_repr =  True
        self.source_head = build_mlp(self.latent_dim, 1024, kwargs['n_cls'])# classification
        self.target_head = build_mlp(self.latent_dim, 1024, kwargs['repr_dim'])
        self.use_target_mask = kwargs.get('use_target_mask', False) 
        self.ablation = kwargs.get('ablation', None)

        self.cimr_aligner = ContinuousImplicitAligner(feature_dim=self.latent_dim) # 维度填你的 D
        # assert self.ablation in ['raw_text', 'no_text', 'raw_source', 'raw_text_and_source'], 'ablation type not supported'
        

    def forward(self,
                noised_motion,
                timestep,
                in_motion_mask,
                text_embeds,
                condition_mask, 
                motion_embeds=None,
                lengths=None,
                src_len = None,
                tgt_len = None,
                **kwargs):
        if isinstance(src_len, list):
            src_len = torch.tensor(src_len)
            tgt_len = torch.tensor(tgt_len)
        # noised_motion: B, T, D
        # timestep: B
        # motion_embeds: T, B, D
        # len: B
        # proj_noised_motion

        # 0.  dimension matching
        # noised_motion [latent_dim[0], batch_size, latent_dim] <= [batch_size, latent_dim[0], latent_dim[1]]
        bs = noised_motion.shape[0]
        noised_motion = noised_motion.permute(1, 0, 2)
        # 0. check lengths for no vae (diffusion only)
        # if lengths not in [None, []]:
        motion_in_mask = in_motion_mask

        # time_embedding | text_embedding | frames_source | frames_target
        # 1 * lat_d | max_text * lat_d | max_frames * lat_d | max_frames * lat_d
        
        
        if self.condition in ["text", "text_uncond"]:
            # make it seq first
            text_embeds = text_embeds.permute(1, 0, 2)
            if self.text_encoded_dim != self.latent_dim:
                # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
                text_emb_latent = self.emb_proj(text_embeds)
            else:
                text_emb_latent = text_embeds
                # source_motion_zeros = torch.zeros(*noised_motion.shape[:2], 
                #                             self.latent_dim, 
                #                             device=noised_motion.device)
                # aux_fake_mask = torch.zeros(condition_mask.shape[0], 
                #                             noised_motion.shape[0], 
                #                             device=noised_motion.device)
                # condition_mask = torch.cat((condition_mask, aux_fake_mask), 
                #                            1).bool().to(noised_motion.device)
            emb_latent_ori = text_emb_latent
            emb_latent = text_emb_latent
            # 1, B, D 

            if motion_embeds is not None:
                zeroes_mask = (motion_embeds == 0).all(dim=-1)
                if motion_embeds.shape[-1] != self.latent_dim:
                    motion_embeds_proj = self.pose_proj_in_source(motion_embeds)
                    motion_embeds_proj[zeroes_mask] = 0
                else:
                    motion_embeds_proj = motion_embeds
 
        else:
            raise TypeError(f"condition type {self.condition} not supported")
        # 4. transformer
        # if self.diffusion_only:
        proj_noised_motion = self.pose_proj_in_target(noised_motion)
        
        # BUILD the mask now
        if motion_embeds is None:
            aug_mask = torch.cat((condition_mask[:, :text_emb_latent.shape[0]],
                                  motion_in_mask), 1)
        else:
            sep_token_mask = torch.ones((bs, self.sep_token.shape[0]),
                                        dtype=bool,
                                        device=noised_motion.device)
            aug_mask = torch.cat((
                            condition_mask[:, text_emb_latent.shape[0]:],
                            sep_token_mask,
                            motion_in_mask,
                            ), 1)

        # NOTE: condition encoding
        # emb_latent: T_Text, B, D
        # motion_embeds_proj: T_max, B, D
        motion_embeds_proj_ori = motion_embeds_proj
        cond_seq = torch.cat((emb_latent, motion_embeds_proj), dim=0)
        cond_seq = self.cond_pos(cond_seq)
        cond_seq_processed = self.cond_encoder(cond_seq, src_key_padding_mask=~condition_mask)
        T_Text = emb_latent.size(0)
        T_max = motion_embeds_proj.size(0)
        emb_latent_processed, motion_embeds_proj_processed = torch.split(cond_seq_processed, [T_Text, T_max], dim=0)
        emb_latent, motion_embeds_proj = emb_latent_processed, motion_embeds_proj_processed

        if motion_embeds is None:
            xseq = proj_noised_motion
        else:      
            sep_token_batch = torch.tile(self.sep_token, (bs,)).reshape(bs,
                                                                        -1)
            if self.ablation in ['raw_source', 'raw_text_and_source']:
                xseq = torch.cat((motion_embeds_proj_ori,
                                sep_token_batch[None],
                                proj_noised_motion), axis=0)
            else:
#---------------------------------------------------------------
                source = motion_embeds_proj.permute(1, 0, 2)
                noised_target = proj_noised_motion.permute(1, 0, 2)
                source = (source - source.mean(dim=1, keepdim=True)) / (source.std(dim=1, keepdim=True) + 1e-8)
                noised_target = (noised_target - noised_target.mean(dim=1, keepdim=True)) / (noised_target.std(dim=1, keepdim=True) + 1e-8)
                warped_source = self.cimr_aligner(source, noised_target.shape[1])
                alpha = 0.05  # 0-1之间，控制source的贡献度
                fusion_target = noised_target + alpha * (warped_source - noised_target)
                proj_noised_motion = fusion_target.permute(1, 0, 2)
#---------------------------------------------------------------                
                xseq = torch.cat((motion_embeds_proj,
                                sep_token_batch[None],
                                proj_noised_motion), axis=0)

        xseq = self.query_pos(xseq)
        if self.use_target_mask:
            mask = None
        else:
            mask = aug_mask
            
        if self.ablation == 'no_text':
            condition = torch.zeros_like(emb_latent[0])
        elif self.ablation == 'raw_text' or self.ablation == 'raw_text_and_source':
            condition = emb_latent_ori[0]
        elif self.ablation == 'raw_source':
            condition = emb_latent[0]
        else:
    # 默认使用 enhanced text + enhanced motion1111111
            condition = emb_latent[0]

        tokens = self.dit_encoder(xseq.permute(1, 0, 2), timestep, condition, mask=mask)
        # B, T, D
        tokens = tokens.permute(1, 0, 2)
        # if self.diffusion_only:
        if motion_embeds is not None:
            denoised_motion_proj = tokens
            if self.use_sep:
                useful_tokens = motion_embeds_proj.shape[0]+1
            else:
                useful_tokens = motion_embeds_proj.shape[0]
            denoised_motion_proj = denoised_motion_proj[useful_tokens:]
        else:
            denoised_motion_proj = tokens

        denoised_motion = self.pose_proj_out(denoised_motion_proj)
        denoised_motion[~motion_in_mask.T] = 0
        # zero for padded area
        # else:
        #     sample = tokens[:sample.shape[0]]
        # 5. [batch_size, latent_dim[0], latent_dim[1]] <= [latent_dim[0], batch_size, latent_dim[1]]
        denoised_motion = denoised_motion.permute(1, 0, 2)
        return denoised_motion

    def forward_with_repr(self,
                noised_motion,
                timestep,
                in_motion_mask,
                text_embeds,
                condition_mask, 
                neg_text_embeds=None,
                motion_embeds=None,
                lengths=None,
                src_len = None,
                tgt_len = None,
                target_align_depth = 6, 
                return_attention = False, 
                **kwargs):
        if isinstance(src_len, list):
            src_len = torch.tensor(src_len)
            tgt_len = torch.tensor(tgt_len)
        # noised_motion: B, T, D
        # timestep: B
        # motion_embeds: T, B, D
        # len: B
        # proj_noised_motion

        # 0.  dimension matching
        # noised_motion [latent_dim[0], batch_size, latent_dim] <= [batch_size, latent_dim[0], latent_dim[1]]
        bs = noised_motion.shape[0]
        noised_motion = noised_motion.permute(1, 0, 2)
        # 0. check lengths for no vae (diffusion only)
        # if lengths not in [None, []]:
        motion_in_mask = in_motion_mask

        # time_embedding | text_embedding | frames_source | frames_target
        # 1 * lat_d | max_text * lat_d | max_frames * lat_d | max_frames * lat_d
        
        
        if self.condition in ["text", "text_uncond"]:
            # make it seq first
            text_embeds = text_embeds.permute(1, 0, 2)
            if self.text_encoded_dim != self.latent_dim:
                # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
                text_emb_latent = self.emb_proj(text_embeds)
                neg_text_emb_latent = self.emb_proj(neg_text_embeds) #@############
            else:
                text_emb_latent = text_embeds
                neg_text_emb_latent = neg_text_embeds #@############+
                # source_motion_zeros = torch.zeros(*noised_motion.shape[:2], 
                #                             self.latent_dim, 
                #                             device=noised_motion.device)
                # aux_fake_mask = torch.zeros(condition_mask.shape[0], 
                #                             noised_motion.shape[0], 
                #                             device=noised_motion.device)
                # condition_mask = torch.cat((condition_mask, aux_fake_mask), 
                #                            1).bool().to(noised_motion.device)
            emb_latent_ori = text_emb_latent
            emb_latent = text_emb_latent
            # 1, B, D 

            if motion_embeds is not None:
                zeroes_mask = (motion_embeds == 0).all(dim=-1)
                if motion_embeds.shape[-1] != self.latent_dim:
                    motion_embeds_proj = self.pose_proj_in_source(motion_embeds)
                    motion_embeds_proj[zeroes_mask] = 0
                else:
                    motion_embeds_proj = motion_embeds
 
        else:
            raise TypeError(f"condition type {self.condition} not supported")
        # 4. transformer
        # if self.diffusion_only:
        proj_noised_motion = self.pose_proj_in_target(noised_motion)
        
        # BUILD the mask now
        if motion_embeds is None:
            aug_mask = torch.cat((condition_mask[:, :text_emb_latent.shape[0]],
                                  motion_in_mask), 1)
        else:
            sep_token_mask = torch.ones((bs, self.sep_token.shape[0]),
                                        dtype=bool,
                                        device=noised_motion.device)
            aug_mask = torch.cat((
                            condition_mask[:, text_emb_latent.shape[0]:],
                            sep_token_mask,
                            motion_in_mask,
                            ), 1)
            # B, T_output

        # NOTE: condition encoding
        # emb_latent: T_Text, B, D
        # motion_embeds_proj: T_max, B, D
        motion_embeds_proj_ori = motion_embeds_proj
        cond_seq = torch.cat((emb_latent, motion_embeds_proj), dim=0)
        cond_seq = self.cond_pos(cond_seq)
        cond_seq_processed = self.cond_encoder(cond_seq, src_key_padding_mask=~condition_mask)
        T_Text = emb_latent.size(0)
        T_max = motion_embeds_proj.size(0)
        emb_latent_processed, motion_embeds_proj_processed = torch.split(cond_seq_processed, [T_Text, T_max], dim=0)
        emb_latent, motion_embeds_proj = emb_latent_processed, motion_embeds_proj_processed

        if motion_embeds is None:
            xseq = proj_noised_motion
        else:      
            sep_token_batch = torch.tile(self.sep_token, (bs,)).reshape(bs,
                                                                        -1)
            if self.ablation in ['raw_source', 'raw_text_and_source']:
                xseq = torch.cat((motion_embeds_proj_ori,
                                sep_token_batch[None],
                                proj_noised_motion), axis=0)
            else:
#---------------------------------------------------------------
                source = motion_embeds_proj.permute(1, 0, 2)
                noised_target = proj_noised_motion.permute(1, 0, 2)
                source = (source - source.mean(dim=1, keepdim=True)) / (source.std(dim=1, keepdim=True) + 1e-8)
                noised_target = (noised_target - noised_target.mean(dim=1, keepdim=True)) / (noised_target.std(dim=1, keepdim=True) + 1e-8)
                warped_source = self.cimr_aligner(source, noised_target.shape[1])
                alpha = 0.05  # 0-1之间，控制source的贡献度
                fusion_target = noised_target + alpha * (warped_source - noised_target)
                proj_noised_motion = fusion_target.permute(1, 0, 2)
#---------------------------------------------------------------                
                xseq = torch.cat((motion_embeds_proj,
                                sep_token_batch[None],
                                proj_noised_motion), axis=0)

        xseq = self.query_pos(xseq)
        if self.use_target_mask:
            mask = None
        else:
            mask = aug_mask
        if self.ablation == 'no_text':
            condition = torch.zeros_like(emb_latent[0])
        elif self.ablation == 'raw_text' or self.ablation == 'raw_text_and_source':
            condition = emb_latent_ori[0]
        elif self.ablation == 'raw_source':
            condition = emb_latent[0]
        else:
    # 默认使用 enhanced text + enhanced motion1111111
            condition = emb_latent[0]

        if return_attention:
            print("------------self.dit_encoder.forward_with_repr_att")
            tokens, target_repr, attention_mask = self.dit_encoder.forward_with_repr_att(xseq.permute(1, 0, 2), \
            timestep, condition, depth=target_align_depth, mask=mask)
        else:
            tokens, target_repr, intermediates = self.dit_encoder.forward_with_repr(xseq.permute(1, 0, 2), \
            timestep, condition, depth=target_align_depth, mask=mask)
        # B, T, D
        tokens = tokens.permute(1, 0, 2)
        # if self.diffusion_only:
        if motion_embeds is not None:
            denoised_motion_proj = tokens
            if self.use_sep:
                useful_tokens = motion_embeds_proj.shape[0]+1
            else:
                useful_tokens = motion_embeds_proj.shape[0]
            denoised_motion_proj = denoised_motion_proj[useful_tokens:]
        else:
            denoised_motion_proj = tokens

        tokens_1 = intermediates[0].permute(1, 0, 2)
        denoised_motion_proj_1 = tokens_1[useful_tokens:]

        tokens_2 = intermediates[1].permute(1, 0, 2)
        denoised_motion_proj_2 = tokens_2[useful_tokens:]

        tokens_3 = intermediates[2].permute(1, 0, 2)
        denoised_motion_proj_3 = tokens_3[useful_tokens:]

        denoised_motion = self.pose_proj_out(denoised_motion_proj)
        denoised_motion[~motion_in_mask.T] = 0
        denoised_motion = denoised_motion.permute(1, 0, 2)

        inter1 = self.inter_proj_1(denoised_motion_proj_1)  # (B,T,207) #@###############
        inter1[~motion_in_mask.T] = 0
        denoised_motion_25 = inter1.permute(1, 0, 2)

        inter2 = self.inter_proj_2(denoised_motion_proj_2)  # (B,T,207)
        inter2[~motion_in_mask.T] = 0
        denoised_motion_50 = inter2.permute(1, 0, 2)

        inter3 = self.inter_proj_3(denoised_motion_proj_3)  # (B,T,207)
        inter3[~motion_in_mask.T] = 0
        denoised_motion_75 = inter3.permute(1, 0, 2)


        target_repr = self.target_head(target_repr[:, T_max+1:]) # B, T_target, D
        source_repr = self.source_head(motion_embeds_proj) # B, T_source, D
        if return_attention:
            return denoised_motion, source_repr.permute(1,0,2), target_repr, attention_mask
        return denoised_motion, source_repr.permute(1,0,2), target_repr, [denoised_motion_25, denoised_motion_50, denoised_motion_75], denoised_motion_proj.permute(1,0,2).mean(dim=1), text_emb_latent.permute(1,0,2).squeeze(1), neg_text_emb_latent.squeeze(1) # 都变为(B,512)

    def forward_with_guidance(self,
                              noised_motion,
                              timestep,
                              in_motion_mask,
                              text_embeds,
                              condition_mask,
                              guidance_motion,
                              guidance_text_n_motion, 
                              motion_embeds=None,
                              lengths=None,
                              inpaint_dict=None,
                              max_steps=None,
                              prob_way='3way',
                              **kwargs):
        # if motion embeds is None
        # TODO put here that you have tow
        # implement 2 cases for that case
        # text unconditional more or less 2 replicas
        # timestep
        if max_steps is not None:
            curr_ts = timestep[0].item()
            g_m = max(1, guidance_motion*2*curr_ts/max_steps)
            guidance_motion = g_m
            g_t_tm = max(1, guidance_text_n_motion*2*curr_ts/max_steps)
            guidance_text_n_motion = g_t_tm

        if motion_embeds is None:
            half = noised_motion[: len(noised_motion) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.forward(combined, timestep,
                                    in_motion_mask=in_motion_mask,
                                    text_embeds=text_embeds,
                                    condition_mask=condition_mask, 
                                    motion_embeds=motion_embeds,
                                    lengths=lengths)
            uncond_eps, cond_eps_text = torch.split(model_out, len(model_out) // 2,
                                                     dim=0)
            # make it BxSxfeatures
            if inpaint_dict is not None:
                import torch.nn.functional as F
                source_mot = inpaint_dict['start_motion'].permute(1, 0, 2)
                if source_mot.shape[1] >= uncond_eps.shape[1]:
                    source_mot = source_mot[:, :uncond_eps.shape[1]]
                else:
                    pad = uncond_eps.shape[1] - source_mot.shape[1]
                    # Pad the tensor on the second dimension (time)
                    source_mot = F.pad(source_mot, (0, 0, 0, pad), 'constant', 0)

                mot_len = source_mot.shape[1]
                # concat mask for all the frames
                mask_src_parts = inpaint_dict['mask'].unsqueeze(1).repeat(1,
                                                                      mot_len,
                                                                      1)
                uncond_eps = uncond_eps*(mask_src_parts) + source_mot*(~mask_src_parts)
                cond_eps_text = cond_eps_text*(mask_src_parts) + source_mot*(~mask_src_parts)
            half_eps = uncond_eps + guidance_text_n_motion * (cond_eps_text - uncond_eps) 
            eps = torch.cat([half_eps, half_eps], dim=0)
        else:
            third = noised_motion[: len(noised_motion) // 3]
            combined = torch.cat([third, third, third], dim=0)
            model_out = self.forward(combined, timestep,
                                     in_motion_mask=in_motion_mask,
                                     text_embeds=text_embeds,
                                     condition_mask=condition_mask, 
                                     motion_embeds=motion_embeds,
                                     lengths=lengths)
            # For exact reproducibility reasons, we apply classifier-free guidance on only
            # three channels by default. The standard approach to cfg applies it to all channels.
            # This can be done by uncommenting the following line and commenting-out the line following that.
            # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
            # eps, rest = model_out[:, :3], model_out[:, 3:]
            uncond_eps, cond_eps_motion, cond_eps_text_n_motion = torch.split(model_out,
                                                                            len(model_out) // 3,
                                                                            dim=0)
            if inpaint_dict is not None:
                import torch.nn.functional as F
                source_mot = inpaint_dict['start_motion'].permute(1, 0, 2)
                if source_mot.shape[1] >= uncond_eps.shape[1]:
                    source_mot = source_mot[:, :uncond_eps.shape[1]]
                else:
                    pad = uncond_eps.shape[1] - source_mot.shape[1]
                    # Pad the tensor on the second dimension (time)
                    source_mot = F.pad(source_mot, (0, 0, 0, pad), 'constant', 0)

                mot_len = source_mot.shape[1]
                # concat mask for all the frames
                mask_src_parts = inpaint_dict['mask'].unsqueeze(1).repeat(1,
                                                                      mot_len,
                                                                      1)
                uncond_eps = uncond_eps*(~mask_src_parts) + source_mot*mask_src_parts
                cond_eps_text = cond_eps_text*(~mask_src_parts) + source_mot*mask_src_parts
                cond_eps_text_n_motion = cond_eps_text_n_motion*(~mask_src_parts) + source_mot*mask_src_parts
            if prob_way=='3way':
                third_eps = uncond_eps + guidance_motion * (cond_eps_motion - uncond_eps) + \
                            guidance_text_n_motion * (cond_eps_text_n_motion - cond_eps_motion)
            if prob_way=='2way':
                third_eps = uncond_eps + guidance_text_n_motion * (cond_eps_text_n_motion - uncond_eps)

            eps = torch.cat([third_eps, third_eps, third_eps], dim=0)
        return eps

import torch

def test_TMED_DiTMotionDenoiser():
    # 设置随机种子以保证测试的可复现性
    torch.manual_seed(42)

    # 测试参数
    batch_size = 8
    seq_length = 100  # 序列长度
    nfeats = 263
    latent_dim = 256
    text_encoded_dim = 768

    # 创建测试输入数据
    noised_motion = torch.randn(batch_size, seq_length, nfeats)  # 输入的噪声运动数据 (N, T, nfeats)
    timestep = torch.randint(0, 1000, (batch_size,))  # 随机生成时间步 (N,)
    text_embeds = torch.randn(batch_size, 1, text_encoded_dim)  # 随机生成文本嵌入 (N, D)
    in_motion_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)  # 假设无填充
    condition_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)  # 条件掩码
    motion_embeds = torch.randn(seq_length, batch_size, nfeats)  # 随机生成运动嵌入 (N, T, latent_dim)

    # 创建模型实例
    model = DiT_Denoiser(
        nfeats=nfeats,
        latent_dim=latent_dim,
        text_encoded_dim=text_encoded_dim,
        num_layers=6,
        num_heads=4,
        dropout=0.1,
        activation='gelu',
        motion_condition = 'source'
    )

    # 将模型设置为评估模式
    model.eval()

    # 前向传播
    with torch.no_grad():
        output = model(noised_motion, timestep, in_motion_mask, text_embeds, condition_mask, motion_embeds=motion_embeds)

    # 检查输出的形状
    expected_shape = (batch_size, seq_length, nfeats)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"

    # 检查输出是否包含合理值 (例如无NaN)
    assert not torch.isnan(output).any(), "Output contains NaN values, which is unexpected."

    print("Test passed: TMED_DiTMotionDenoiser handles 1D inputs correctly!")

# 运行测试
if __name__ == "__main__":
    test_TMED_DiTMotionDenoiser()
