import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
from typing import List, Tuple, Optional, Dict, Any

from .fractal_curve_tokenizer import FractalHilbertTokenizer

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def create_attention_mask(levels_info: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    """根据层级信息创建注意力掩码，同层级token之间的注意力更强"""
    max_len = max(len(info) for info in levels_info)
    batch_size = len(levels_info)
    
    mask = torch.ones(batch_size, max_len, max_len, device=device)
    
    for b, level_info in enumerate(levels_info):
        seq_len = len(level_info)
        if seq_len == 0:
            continue
            
        for i in range(seq_len):
            for j in range(seq_len):
                if i < len(level_info) and j < len(level_info):
                    # 同层级token之间的注意力权重更高
                    level_i = level_info[i, 0].item() if level_info[i].numel() > 0 else 0
                    level_j = level_info[j, 0].item() if level_info[j].numel() > 0 else 0
                    
                    if level_i == level_j:
                        mask[b, i, j] = 1.2  # 同层级增强
                    elif abs(level_i - level_j) == 1:
                        mask[b, i, j] = 1.1  # 相邻层级稍微增强
                    else:
                        mask[b, i, j] = 1.0  # 正常权重
    
    return mask

# classes

class AdvancedFractalPositionEmbedding(nn.Module):
    """
    高级分形位置编码，完全对齐增强tokenizer
    支持动态层级、Hilbert路径编码和多尺度空间感知
    """
    
    def __init__(self, dim: int, max_level: int = 50, max_seq_len: int = 10000, 
                 use_hilbert_encoding: bool = True, use_spatial_encoding: bool = True):
        super().__init__()
        self.dim = dim
        self.max_level = max_level
        self.max_seq_len = max_seq_len
        self.use_hilbert_encoding = use_hilbert_encoding
        self.use_spatial_encoding = use_spatial_encoding
        
        # 层级深度嵌入 - 支持最多50层
        self.depth_embedding = nn.Embedding(max_level + 1, dim)
        
        # Hilbert路径编码 - 编码分形路径信息
        if use_hilbert_encoding:
            self.path_embedding = nn.Embedding(max_seq_len, dim)
            # 路径序列编码器（RNN-based）
            self.path_encoder = nn.LSTM(
                input_size=dim, 
                hidden_size=dim // 2, 
                num_layers=2, 
                batch_first=True,
                bidirectional=True
            )
        
        # 多尺度空间位置编码
        if use_spatial_encoding:
            # 可学习的2D空间嵌入
            self.spatial_embedding_2d = nn.Parameter(torch.randn(1, max_seq_len, dim))
            # 尺度感知的位置编码
            self.scale_embedding = nn.Embedding(max_level + 1, dim)
        
        # 特征融合网络
        input_dim = dim  # depth_embedding
        if use_hilbert_encoding:
            input_dim += dim  # path encoding
        if use_spatial_encoding:
            input_dim += dim * 2  # spatial + scale
            
        self.fusion_network = nn.Sequential(
            nn.Linear(input_dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
        
        # 自适应权重，平衡不同类型编码的贡献
        self.encoding_weights = nn.Parameter(torch.ones(4))  # [depth, path, spatial, scale]
        
        # 层级感知的注意力偏置
        self.level_attention_bias = nn.Parameter(torch.zeros(max_level + 1, max_level + 1))
        
        # 初始化
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化参数"""
        # 使用正态分布初始化嵌入
        nn.init.normal_(self.depth_embedding.weight, std=0.02)
        if self.use_hilbert_encoding:
            nn.init.normal_(self.path_embedding.weight, std=0.02)
        if self.use_spatial_encoding:
            nn.init.normal_(self.spatial_embedding_2d, std=0.02)
            nn.init.normal_(self.scale_embedding.weight, std=0.02)
            
        # 初始化层级注意力偏置为小值
        nn.init.uniform_(self.level_attention_bias, -0.1, 0.1)
    
    def forward(self, levels_info: torch.Tensor, sequence_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            levels_info: [N, max_info_len] 层级信息矩阵，第一列是depth，其余是path
            sequence_positions: [N] 序列中的位置索引（可选）
            
        Returns:
            position_emb: [N, dim] 位置编码
        """
        if levels_info.numel() == 0:
            return torch.zeros(0, self.dim, device=levels_info.device, dtype=torch.float32)
            
        device = levels_info.device
        N = levels_info.shape[0]
        
        # 提取深度信息
        depths = levels_info[:, 0].clamp(0, self.max_level)  # [N]
        
        # 1. 深度编码
        depth_emb = self.depth_embedding(depths)  # [N, dim]
        
        embeddings = [depth_emb]
        
        # 2. Hilbert路径编码
        if self.use_hilbert_encoding and levels_info.shape[1] > 1:
            path_info = levels_info[:, 1:]  # [N, path_len]
            
            # 将路径信息转换为嵌入序列
            path_embeddings = []
            for i in range(path_info.shape[1]):
                path_indices = path_info[:, i].clamp(0, self.max_seq_len - 1)
                path_emb = self.path_embedding(path_indices)  # [N, dim]
                path_embeddings.append(path_emb)
            
            if path_embeddings:
                path_sequence = torch.stack(path_embeddings, dim=1)  # [N, path_len, dim]
                
                # 使用LSTM编码路径序列
                path_encoded, _ = self.path_encoder(path_sequence)  # [N, path_len, dim]
                # 取最后一个时间步的输出
                path_final = path_encoded[:, -1, :]  # [N, dim]
                embeddings.append(path_final)
            else:
                embeddings.append(torch.zeros_like(depth_emb))
        
        # 3. 空间位置编码
        if self.use_spatial_encoding:
            if sequence_positions is not None:
                # 使用提供的序列位置
                seq_pos = sequence_positions.clamp(0, self.max_seq_len - 1)
            else:
                # 使用默认序列位置
                seq_pos = torch.arange(N, device=device).clamp(0, self.max_seq_len - 1)
            
            # 空间嵌入
            spatial_emb = self.spatial_embedding_2d[0, seq_pos, :]  # [N, dim]
            embeddings.append(spatial_emb)
            
            # 尺度嵌入
            scale_emb = self.scale_embedding(depths)  # [N, dim]
            embeddings.append(scale_emb)
        
        # 4. 特征融合
        combined_emb = torch.cat(embeddings, dim=-1)  # [N, total_dim]
        fused_emb = self.fusion_network(combined_emb)  # [N, dim]
        
        # 5. 应用自适应权重
        if len(embeddings) == 4:  # 所有编码都启用
            weighted_emb = (
                embeddings[0] * self.encoding_weights[0] +
                embeddings[1] * self.encoding_weights[1] +
                embeddings[2] * self.encoding_weights[2] +
                embeddings[3] * self.encoding_weights[3]
            ) / self.encoding_weights.sum()
        else:
            weighted_emb = fused_emb
        
        # 6. 残差连接
        final_emb = fused_emb + weighted_emb * 0.1
        
        return final_emb
    
    def get_attention_bias(self, depths: torch.Tensor) -> torch.Tensor:
        """获取基于层级的注意力偏置"""
        N = len(depths)
        bias = torch.zeros(N, N, device=depths.device)
        
        for i in range(N):
            for j in range(N):
                depth_i = depths[i].clamp(0, self.max_level).long()
                depth_j = depths[j].clamp(0, self.max_level).long()
                bias[i, j] = self.level_attention_bias[depth_i, depth_j]
        
        return bias


class EnhancedFractalTokenProcessor(nn.Module):
    """
    增强的分形token处理器，对齐tokenizer的特征提取能力
    支持多尺度特征、边缘信息和纹理分析
    """
    
    def __init__(self, input_dim: int, output_dim: int, min_patch_size: Tuple[int, int] = (4, 4), 
                 max_level: int = 50, use_feature_enhancement: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.min_patch_size = min_patch_size
        self.max_level = max_level
        self.use_feature_enhancement = use_feature_enhancement
        
        # 基础token投影
        self.token_norm = nn.LayerNorm(input_dim)
        self.token_projection = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim)
        )
        
        # 层级类型嵌入
        self.level_type_embedding = nn.Embedding(max_level + 1, output_dim)
        
        # 特征增强网络（模拟tokenizer的特征提取）
        if use_feature_enhancement:
            # 统计特征处理器
            self.stats_processor = nn.Sequential(
                nn.Linear(2, output_dim // 4),  # 输入：方差和均值
                nn.ReLU(),
                nn.Linear(output_dim // 4, output_dim // 4)
            )
            
            # 边缘特征处理器
            self.edge_processor = nn.Sequential(
                nn.Linear(1, output_dim // 4),  # 输入：边缘密度
                nn.ReLU(),
                nn.Linear(output_dim // 4, output_dim // 4)
            )
            
            # 空间特征处理器
            self.spatial_processor = nn.Sequential(
                nn.Linear(2, output_dim // 4),  # 输入：高度和宽度
                nn.ReLU(),
                nn.Linear(output_dim // 4, output_dim // 4)
            )
            
            # 层级特征处理器
            self.level_processor = nn.Sequential(
                nn.Linear(1, output_dim // 4),  # 输入：层级
                nn.ReLU(),
                nn.Linear(output_dim // 4, output_dim // 4)
            )
            
            # 特征融合网络
            self.feature_fusion = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        
        # 多尺度适配器
        self.scale_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            ) for _ in range(max_level + 1)
        ])
        
        # 动态权重网络
        self.dynamic_weighting = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def _extract_token_features(self, tokens: torch.Tensor, level: int, 
                               patch_height: int, patch_width: int) -> Dict[str, torch.Tensor]:
        """提取token的增强特征，模拟tokenizer的特征提取"""
        device = tokens.device
        batch_size = tokens.shape[0]
        
        # 基础统计特征
        token_var = torch.var(tokens, dim=-1, keepdim=True)  # [N, 1]
        token_mean = torch.mean(tokens, dim=-1, keepdim=True)  # [N, 1]
        
        # 模拟边缘特征（基于token值的变化率）
        if tokens.shape[-1] > 1:
            token_diff = torch.diff(tokens, dim=-1)
            edge_density = torch.mean(torch.abs(token_diff), dim=-1, keepdim=True)  # [N, 1]
        else:
            edge_density = torch.zeros(batch_size, 1, device=device)
        
        # 空间特征
        spatial_features = torch.tensor(
            [[patch_height, patch_width]], 
            device=device, dtype=torch.float32
        ).expand(batch_size, -1)  # [N, 2]
        
        # 层级特征
        level_features = torch.tensor(
            [[level]], device=device, dtype=torch.float32
        ).expand(batch_size, -1)  # [N, 1]
        
        return {
            'stats': torch.cat([token_var, token_mean], dim=-1),  # [N, 2]
            'edge': edge_density,  # [N, 1]
            'spatial': spatial_features,  # [N, 2]
            'level': level_features  # [N, 1]
        }
    
    def forward(self, tokens_list: List[torch.Tensor], levels_list: List[torch.Tensor], 
                patch_info_list: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            tokens_list: List[Tensor] 每个图像的token列表 [N_i, input_dim]
            levels_list: List[Tensor] 每个图像的层级信息列表 [N_i, max_info_len]
            patch_info_list: List[Dict] 每个图像的patch信息（可选）
            
        Returns:
            processed_tokens: List[Tensor] 处理后的token [N_i, output_dim]
            enhanced_levels: List[Tensor] 增强的层级信息 [N_i, max_info_len]
        """
        processed_tokens = []
        enhanced_levels = []
        
        for i, (tokens, levels) in enumerate(zip(tokens_list, levels_list)):
            if tokens.numel() == 0:
                processed_tokens.append(torch.empty(0, self.output_dim, device=tokens.device))
                enhanced_levels.append(levels)
                continue
                
            # 基础处理
            norm_tokens = self.token_norm(tokens)  # [N, input_dim]
            proj_tokens = self.token_projection(norm_tokens)  # [N, output_dim]
            
            # 添加层级类型嵌入
            if levels.numel() > 0 and levels.shape[0] > 0:
                depths = levels[:, 0].clamp(0, self.max_level)
                type_emb = self.level_type_embedding(depths)  # [N, output_dim]
                proj_tokens = proj_tokens + type_emb
            
            # 特征增强
            if self.use_feature_enhancement and levels.numel() > 0:
                # 获取patch信息
                patch_info = patch_info_list[i] if patch_info_list else {}
                patch_height = patch_info.get('height', self.min_patch_size[0])
                patch_width = patch_info.get('width', self.min_patch_size[1])
                level = int(levels[0, 0].item()) if levels.shape[0] > 0 else 0
                
                # 提取增强特征
                features = self._extract_token_features(tokens, level, patch_height, patch_width)
                
                # 处理各种特征
                stats_feat = self.stats_processor(features['stats'])  # [N, dim//4]
                edge_feat = self.edge_processor(features['edge'])  # [N, dim//4]
                spatial_feat = self.spatial_processor(features['spatial'])  # [N, dim//4]
                level_feat = self.level_processor(features['level'])  # [N, dim//4]
                
                # 拼接增强特征
                enhanced_feat = torch.cat([stats_feat, edge_feat, spatial_feat, level_feat], dim=-1)  # [N, dim]
                enhanced_feat = self.feature_fusion(enhanced_feat)  # [N, dim]
                
                # 融合基础投影和增强特征
                proj_tokens = proj_tokens + enhanced_feat * 0.3
            
            # 多尺度自适应
            if levels.numel() > 0 and levels.shape[0] > 0:
                final_tokens = []
                for j, token in enumerate(proj_tokens):
                    if j < len(levels):
                        level_idx = int(levels[j, 0].clamp(0, self.max_level).item())
                        adapted_token = self.scale_adapters[level_idx](token.unsqueeze(0)).squeeze(0)  # [dim]
                    else:
                        adapted_token = token
                    final_tokens.append(adapted_token)
                
                final_tokens = torch.stack(final_tokens)  # [N, output_dim]
            else:
                final_tokens = proj_tokens
            
            # 动态权重调整
            weights = self.dynamic_weighting(final_tokens)  # [N, 1]
            final_tokens = final_tokens * (0.8 + 0.4 * weights)  # 权重范围[0.8, 1.2]
            
            processed_tokens.append(final_tokens)
            enhanced_levels.append(levels)
            
        return processed_tokens, enhanced_levels


class HilbertAwareMultiScaleAttention(nn.Module):
    """
    Hilbert曲线感知的多尺度注意力机制
    结合tokenizer的路径编码和层级信息
    """
    
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0., 
                 max_level: int = 50, use_hilbert_bias: bool = True, use_level_scaling: bool = True):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.max_level = max_level
        self.use_hilbert_bias = use_hilbert_bias
        self.use_level_scaling = use_level_scaling
        
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        # Hilbert路径感知的注意力偏置
        if use_hilbert_bias:
            self.hilbert_bias_network = nn.Sequential(
                nn.Linear(2, 64),  # 输入：两个token的路径差异特征
                nn.ReLU(),
                nn.Linear(64, heads),  # 输出每个头的偏置
                nn.Tanh()
            )
        
        # 层级感知的缩放因子
        if use_level_scaling:
            self.level_scale_embedding = nn.Embedding(max_level + 1, heads)
            # 初始化为接近1的值
            nn.init.constant_(self.level_scale_embedding.weight, 1.0)
            nn.init.normal_(self.level_scale_embedding.weight, std=0.1)
        
        # 多尺度注意力权重
        self.scale_weights = nn.Parameter(torch.ones(heads))
        
        # 相对位置编码（基于Hilbert距离）
        self.relative_pos_embedding = nn.Embedding(2 * max_level + 1, heads)
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
    def _compute_hilbert_bias(self, levels_info: torch.Tensor) -> Optional[torch.Tensor]:
        """计算基于Hilbert路径的注意力偏置"""
        if not self.use_hilbert_bias or levels_info.numel() == 0:
            return None
            
        seq_len = levels_info.shape[0]
        device = levels_info.device
        
        # 提取路径信息（假设第2列开始是路径）
        if levels_info.shape[1] > 1:
            paths = levels_info[:, 1:].float()  # [seq_len, path_len]
        else:
            return None
        
        # 计算成对的路径差异
        bias_matrix = torch.zeros(self.heads, seq_len, seq_len, device=device)
        
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:
                    # 计算路径差异特征
                    path_i = paths[i]
                    path_j = paths[j]
                    
                    # 路径欧几里得距离
                    path_dist = torch.norm(path_i - path_j).unsqueeze(0)  # [1]
                    # 路径余弦相似度
                    path_sim = F.cosine_similarity(path_i.unsqueeze(0), path_j.unsqueeze(0), dim=1)  # [1]
                    
                    # 组合特征
                    path_features = torch.cat([path_dist, path_sim])  # [2]
                    
                    # 计算偏置
                    bias = self.hilbert_bias_network(path_features)  # [heads]
                    bias_matrix[:, i, j] = bias
        
        return bias_matrix
    
    def _compute_level_bias(self, levels_info: torch.Tensor) -> Optional[torch.Tensor]:
        """计算基于层级的相对位置偏置"""
        if levels_info.numel() == 0:
            return None
            
        seq_len = levels_info.shape[0]
        depths = levels_info[:, 0]  # [seq_len]
        
        # 计算层级差异矩阵
        level_diff = depths.unsqueeze(0) - depths.unsqueeze(1)  # [seq_len, seq_len]
        level_diff = level_diff.clamp(-self.max_level, self.max_level) + self.max_level  # 偏移到正数范围
        
        # 获取相对位置编码
        rel_pos_bias = self.relative_pos_embedding(level_diff)  # [seq_len, seq_len, heads]
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1)  # [heads, seq_len, seq_len]
        
        return rel_pos_bias
    
    def forward(self, x: torch.Tensor, levels_info: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
            levels_info: [seq_len, max_info_len] 层级信息
            attention_mask: [batch, seq_len, seq_len] 注意力掩码
        """
        batch, seq_len, dim = x.shape
        
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # 基础注意力计算
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # 应用多尺度权重
        scale_weights = self.scale_weights.view(1, -1, 1, 1)
        dots = dots * scale_weights
        
        # 应用层级感知的缩放
        if self.use_level_scaling and levels_info is not None and levels_info.numel() > 0:
            depths = levels_info[:, 0].clamp(0, self.max_level)
            level_scales = self.level_scale_embedding(depths)  # [seq_len, heads]
            level_scales = level_scales.transpose(0, 1).unsqueeze(0).unsqueeze(-1)  # [1, heads, seq_len, 1]
            dots = dots * level_scales
        
        # 添加Hilbert路径偏置
        if levels_info is not None:
            hilbert_bias = self._compute_hilbert_bias(levels_info)
            if hilbert_bias is not None:
                dots = dots + hilbert_bias.unsqueeze(0) * 0.1  # 小的偏置影响
            
            # 添加层级相对位置偏置
            level_bias = self._compute_level_bias(levels_info)
            if level_bias is not None:
                dots = dots + level_bias.unsqueeze(0) * 0.05  # 更小的偏置影响
        
        # 应用注意力掩码
        if attention_mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            dots.masked_fill_(~attention_mask.bool(), mask_value)
        
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class AdaptiveFractalFeedForward(nn.Module):
    """
    自适应分形前馈网络，根据tokenizer提取的特征动态调整
    """
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0., 
                 max_level: int = 50, use_level_adaptation: bool = True,
                 use_feature_gating: bool = True):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.max_level = max_level
        self.use_level_adaptation = use_level_adaptation
        self.use_feature_gating = use_feature_gating
        
        self.norm = nn.LayerNorm(dim)
        
        # 主前馈网络
        self.main_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # 层级自适应分支
        if use_level_adaptation:
            self.level_adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, dim),
                    nn.Dropout(dropout)
                ) for _ in range(max_level + 1)
            ])
            
            # 层级混合权重
            self.level_mixing_weights = nn.Parameter(torch.ones(max_level + 1))
        
        # 特征门控网络
        if use_feature_gating:
            self.feature_gate = nn.Sequential(
                nn.Linear(dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, hidden_dim),
                nn.Sigmoid()
            )
        
        # 动态激活函数选择
        self.activation_selector = nn.Sequential(
            nn.Linear(dim, 3),  # 选择GELU, ReLU, Swish
            nn.Softmax(dim=-1)
        )
        
    def _apply_dynamic_activation(self, x: torch.Tensor, activation_weights: torch.Tensor) -> torch.Tensor:
        """应用动态选择的激活函数"""
        gelu_out = F.gelu(x)
        relu_out = F.relu(x)
        swish_out = x * torch.sigmoid(x)
        
        # 加权组合
        output = (activation_weights[:, :, 0:1] * gelu_out + 
                 activation_weights[:, :, 1:2] * relu_out + 
                 activation_weights[:, :, 2:3] * swish_out)
        
        return output
    
    def forward(self, x: torch.Tensor, levels_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
            levels_info: [seq_len, max_info_len] 层级信息
        """
        batch, seq_len, dim = x.shape
        x_norm = self.norm(x)
        
        # 主前馈分支
        main_out = self.main_net(x_norm)
        
        # 层级自适应分支
        if self.use_level_adaptation and levels_info is not None and levels_info.numel() > 0:
            depths = levels_info[:, 0].clamp(0, self.max_level)
            
            # 为每个位置应用对应层级的适配器
            level_outputs = []
            for i, depth in enumerate(depths):
                depth_idx = int(depth.item())
                if i < seq_len:
                    token_input = x_norm[:, i:i+1, :]  # [batch, 1, dim]
                    level_out = self.level_adapters[depth_idx](token_input)  # [batch, 1, dim]
                    level_outputs.append(level_out)
            
            if level_outputs:
                level_adapted = torch.cat(level_outputs, dim=1)  # [batch, seq_len, dim]
                
                # 计算混合权重
                mixing_weights = F.softmax(self.level_mixing_weights[depths], dim=0)
                mixing_weights = mixing_weights.view(1, seq_len, 1)
                
                # 混合主输出和层级适配输出
                main_out = main_out * (1 - mixing_weights) + level_adapted * mixing_weights
        
        # 特征门控
        if self.use_feature_gating:
            gates = self.feature_gate(x_norm)  # [batch, seq_len, hidden_dim]
            
            # 将门控应用到中间层
            # 重新计算主分支以应用门控
            hidden = F.linear(x_norm, self.main_net[0].weight, self.main_net[0].bias)  # [batch, seq_len, hidden_dim]
            gated_hidden = hidden * gates
            
            # 选择动态激活函数
            activation_weights = self.activation_selector(x_norm)  # [batch, seq_len, 3]
            activated_hidden = self._apply_dynamic_activation(gated_hidden, activation_weights)
            
            # 完成前馈 (main_net[3] 是第二个线性层, main_net[4] 是最后的dropout)
            main_out = F.linear(activated_hidden, self.main_net[3].weight, self.main_net[3].bias)
            main_out = self.main_net[4](main_out)  # dropout
        
        return main_out


class EnhancedFractalTransformerBlock(nn.Module):
    """
    增强的分形Transformer块，完全对齐最新tokenizer特性
    """
    
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, 
                 dropout: float = 0., max_level: int = 50):
        super().__init__()
        self.dim = dim
        self.max_level = max_level
        
        # 使用增强的注意力机制
        self.attention = HilbertAwareMultiScaleAttention(
            dim=dim, 
            heads=heads, 
            dim_head=dim_head, 
            dropout=dropout,
            max_level=max_level
        )
        
        # 使用自适应前馈网络
        self.ff = AdaptiveFractalFeedForward(
            dim=dim, 
            hidden_dim=mlp_dim, 
            dropout=dropout,
            max_level=max_level
        )
        
        # 自适应残差权重
        self.residual_weights = nn.Parameter(torch.ones(2))
        
        # 层级感知的层归一化
        self.level_aware_norm1 = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(max_level + 1)
        ])
        self.level_aware_norm2 = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(max_level + 1)
        ])
        
        # 默认层归一化（兜底）
        self.default_norm1 = nn.LayerNorm(dim)
        self.default_norm2 = nn.LayerNorm(dim)
        
    def _apply_level_aware_norm(self, x: torch.Tensor, levels_info: Optional[torch.Tensor], 
                               norm_layers: nn.ModuleList, default_norm: nn.LayerNorm) -> torch.Tensor:
        """应用层级感知的层归一化"""
        if levels_info is None or levels_info.numel() == 0:
            return default_norm(x)
        
        batch_size, seq_len, dim = x.shape
        output = torch.zeros_like(x)
        
        depths = levels_info[:, 0].clamp(0, self.max_level)
        
        for i in range(seq_len):
            level_idx = int(depths[i].item())
            token = x[:, i:i+1, :]  # [batch, 1, dim]
            normed_token = norm_layers[level_idx](token)
            output[:, i:i+1, :] = normed_token
            
        return output
        
    def forward(self, x: torch.Tensor, levels_info: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
            levels_info: [seq_len, max_info_len] 层级信息
            attention_mask: [batch, seq_len, seq_len] 注意力掩码
        """
        # 第一个子层：层级感知归一化 + 注意力
        norm1_x = self._apply_level_aware_norm(x, levels_info, self.level_aware_norm1, self.default_norm1)
        attn_out = self.attention(norm1_x, levels_info, attention_mask)
        x = x + attn_out * self.residual_weights[0]
        
        # 第二个子层：层级感知归一化 + 前馈
        norm2_x = self._apply_level_aware_norm(x, levels_info, self.level_aware_norm2, self.default_norm2)
        ff_out = self.ff(norm2_x, levels_info)
        x = x + ff_out * self.residual_weights[1]
        
        return x


class EnhancedFractalTransformer(nn.Module):
    """
    增强的分形Transformer，完全对齐tokenizer特性
    支持动态层级、Hilbert路径编码和多尺度处理
    """
    
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, 
                 mlp_dim: int, dropout: float = 0., max_level: int = 50):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.max_level = max_level
        
        # 构建增强的Transformer层
        self.layers = nn.ModuleList([
            EnhancedFractalTransformerBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                dropout=dropout,
                max_level=max_level
            ) for _ in range(depth)
        ])
        
        # 全局上下文聚合器（跨层级信息融合）
        self.global_context_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 层级信息聚合
        self.level_aggregator = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
            nn.LayerNorm(dim)
        )
        
        # 最终层归一化
        self.final_norm = nn.LayerNorm(dim)
        
        # 动态深度选择（可选）
        self.depth_selector = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(dim, depth),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, levels_info: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, 
                use_dynamic_depth: bool = False) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
            levels_info: [seq_len, max_info_len] 层级信息  
            attention_mask: [batch, seq_len, seq_len] 注意力掩码
            use_dynamic_depth: 是否使用动态深度选择
        """
        batch_size, seq_len, dim = x.shape
        
        # 动态深度选择
        layer_weights = None
        if use_dynamic_depth:
            # 计算每层的重要性权重
            pooled = x.transpose(1, 2)  # [batch, dim, seq_len]
            layer_weights = self.depth_selector(pooled)  # [batch, depth]
        
        # 逐层处理
        layer_outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, levels_info, attention_mask)
            
            # 应用动态权重（如果启用）
            if layer_weights is not None:
                weight = layer_weights[:, i:i+1].unsqueeze(-1)  # [batch, 1, 1]
                x = x * weight
            
            layer_outputs.append(x)
        
        # 全局上下文聚合（可选）
        if seq_len > 1:
            global_context, _ = self.global_context_attn(x, x, x)
            x = x + global_context * 0.1  # 轻微的全局调整
        
        # 层级信息聚合
        if levels_info is not None and levels_info.numel() > 0:
            # 基于层级信息对输出进行聚合
            aggregated = self.level_aggregator(x)
            x = x + aggregated * 0.2
        
        # 最终归一化
        x = self.final_norm(x)
        
        return x

# 下一代分形ViT - 完全对齐增强tokenizer
class NextGenerationFractalViT(nn.Module):
    """
    下一代分形ViT，完全对齐增强的FractalHilbertTokenizer特性：
    
    核心特性：
    - 可学习的分割决策网络（6特征输入）
    - 真正的Hilbert曲线递归算法（支持多方向）
    - 动态层级管理（最大50层）
    - 智能patch处理和特征增强
    - 边缘检测和纹理复杂度分析
    - 自适应多尺度处理
    """
    
    def __init__(
        self, 
        *, 
        image_size: int, 
        num_classes: int, 
        dim: int = 512, 
        depth: int = 6, 
        heads: int = 8, 
        mlp_dim: int = 1024, 
        pool: str = 'cls', 
        channels: int = 3, 
        dim_head: int = 64, 
        dropout: float = 0., 
        emb_dropout: float = 0.,
        min_patch_size: Tuple[int, int] = (4, 4), 
        max_level: int = 50,  # 对齐tokenizer的最大层级
        learnable_split: bool = True,
        adaptive_threshold: float = 0.5,
        use_hilbert_encoding: bool = True,
        use_spatial_encoding: bool = True,
        use_feature_enhancement: bool = True,
        use_dynamic_depth: bool = False
    ):
        super().__init__()
        
        self.image_size = pair(image_size)
        self.num_classes = num_classes
        self.dim = dim
        self.pool = pool
        self.max_level = max_level
        self.use_dynamic_depth = use_dynamic_depth
        
        # 增强的分形tokenizer（与最新版本完全对齐）
        self.fractal_tokenizer = FractalHilbertTokenizer(
            min_patch_size=min_patch_size,
            max_level=max_level,
            learnable_split=learnable_split,
            adaptive_threshold=adaptive_threshold
        )
        
        # 计算token维度
        patch_dim = channels * min_patch_size[0] * min_patch_size[1]
        
        # 增强的token处理器
        self.token_processor = EnhancedFractalTokenProcessor(
            input_dim=patch_dim,
            output_dim=dim,
            min_patch_size=min_patch_size,
            max_level=max_level,
            use_feature_enhancement=use_feature_enhancement
        )
        
        # 高级分形位置编码
        self.pos_embedding = AdvancedFractalPositionEmbedding(
            dim=dim,
            max_level=max_level,
            max_seq_len=10000,
            use_hilbert_encoding=use_hilbert_encoding,
            use_spatial_encoding=use_spatial_encoding
        )
        
        # CLS token和dropout
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # 增强的分形Transformer
        self.transformer = EnhancedFractalTransformer(
            dim=dim, 
            depth=depth, 
            heads=heads, 
            dim_head=dim_head, 
            mlp_dim=mlp_dim, 
            dropout=dropout,
            max_level=max_level
        )
        
        # 分类头
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, num_classes)
        )
        
        # 层级权重（动态平衡不同层级token的贡献）
        self.level_weights = nn.Parameter(torch.ones(max_level + 1))
        
        # 可学习的池化策略选择
        self.pooling_selector = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(dim, 2),  # CLS vs Mean pooling
            nn.Softmax(dim=-1)
        )
        
        # 辅助损失权重（用于可学习分割决策的正则化）
        self.aux_loss_weight = nn.Parameter(torch.tensor(0.1))
        
        # 特征分析器（用于调试和可视化）
        self.feature_analyzer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 6)  # 输出6个分析指标
        )
        
    def _create_attention_mask(self, levels_list: List[torch.Tensor], max_len: int, device: torch.device) -> torch.Tensor:
        """创建基于层级信息的注意力掩码"""
        batch_size = len(levels_list)
        mask = torch.ones(batch_size, max_len, max_len, device=device)
        
        for b, levels in enumerate(levels_list):
            if levels.numel() == 0:
                continue
                
            seq_len = len(levels)
            depths = levels[:, 0] if levels.numel() > 0 else torch.zeros(seq_len, device=device)
            
            for i in range(seq_len):
                for j in range(seq_len):
                    # 确保索引不会越界 (CLS token + patches)
                    mask_i = i + 1  # +1 for CLS token
                    mask_j = j + 1  # +1 for CLS token
                    
                    if (i < len(depths) and j < len(depths) and 
                        mask_i < max_len and mask_j < max_len):
                        level_i = depths[i].item()
                        level_j = depths[j].item()
                        
                        # 同层级token之间的注意力增强
                        if level_i == level_j:
                            mask[b, mask_i, mask_j] = 1.3  # 同层级增强
                        elif abs(level_i - level_j) == 1:
                            mask[b, mask_i, mask_j] = 1.1  # 邻近层级
                        else:
                            mask[b, mask_i, mask_j] = 1.0  # 基础注意力
        
        return mask
    
    def forward(self, img: torch.Tensor, return_attention: bool = False, 
                return_aux_info: bool = False, return_features: bool = False):
        """
        Args:
            img: [B, C, H, W] 输入图像
            return_attention: 是否返回注意力权重
            return_aux_info: 是否返回辅助信息
            return_features: 是否返回特征分析
            
        Returns:
            logits: [B, num_classes] 分类logits
            其他信息（可选）
        """
        batch_size = img.shape[0]
        device = img.device
        
        # 1. 分形tokenization
        tokens_list, levels_list = self.fractal_tokenizer(img)
        
        # 2. 批量处理
        batch_outputs = []
        aux_infos = []
        features_list = []
        
        for b in range(batch_size):
            tokens = tokens_list[b]
            levels = levels_list[b]
            
            # 处理空token情况
            if tokens.numel() == 0 or tokens.shape[0] == 0:
                batch_outputs.append(torch.zeros(1, self.num_classes, device=device))
                if return_aux_info:
                    aux_infos.append({'num_tokens': 0, 'levels_used': [], 'token_distribution': torch.zeros(self.max_level + 1)})
                if return_features:
                    features_list.append(torch.zeros(6, device=device))
                continue
            
            # 3. Token处理和投影
            processed_tokens, enhanced_levels = self.token_processor([tokens], [levels])
            x = processed_tokens[0].unsqueeze(0)  # [1, N, dim]
            level_info = enhanced_levels[0]  # [N, max_info_len]
            
            # 4. 位置编码
            if level_info.numel() > 0:
                # 创建序列位置索引
                seq_positions = torch.arange(x.shape[1], device=device)
                pos_emb = self.pos_embedding(level_info, seq_positions)  # [N, dim]
                x = x + pos_emb.unsqueeze(0)  # [1, N, dim] + [1, N, dim]
            
            # 5. 添加CLS token
            cls_tokens = self.cls_token.expand(1, -1, -1)  # [1, 1, dim]
            x = torch.cat((cls_tokens, x), dim=1)  # [1, 1+N, dim]
            
            # 更新level_info以包含CLS token的层级信息
            if level_info.numel() > 0:
                cls_level = torch.zeros(1, level_info.shape[1], device=device, dtype=level_info.dtype)
                level_info = torch.cat([cls_level, level_info], dim=0)  # [1+N, max_info_len]
            
            x = self.dropout(x)
            
            # 6. 创建注意力掩码
            attention_mask = None
            if level_info.numel() > 0:
                attention_mask = self._create_attention_mask([level_info], x.shape[1], device)
            
            # 7. Transformer处理
            x = self.transformer(x, level_info, attention_mask, self.use_dynamic_depth)  # [1, 1+N, dim]
            
            # 8. 池化策略选择
            if self.pool == 'cls':
                pooled = x[:, 0].squeeze(0)  # [dim] - CLS token
            elif self.pool == 'mean':
                if x.shape[1] > 1:
                    # 使用层级权重进行加权平均
                    if level_info.numel() > 0 and level_info.shape[0] > 1:
                        token_levels = level_info[1:, 0].clamp(0, self.max_level)  # 排除CLS token
                        weights = F.softmax(self.level_weights[token_levels], dim=0)
                        pooled = torch.sum(x[0, 1:] * weights.unsqueeze(-1), dim=0)  # [dim]
                    else:
                        pooled = x[:, 1:].mean(dim=1).squeeze(0)  # [dim]
                else:
                    pooled = x[:, 0].squeeze(0)  # 回退到CLS
            else:  # 动态选择
                pooling_weights = self.pooling_selector(x.transpose(1, 2))  # [1, 2]
                cls_pooled = x[:, 0]  # [1, dim]
                mean_pooled = x[:, 1:].mean(dim=1) if x.shape[1] > 1 else cls_pooled  # [1, dim]
                pooled = (pooling_weights[0, 0] * cls_pooled + pooling_weights[0, 1] * mean_pooled).squeeze(0)
            
            pooled = self.to_latent(pooled)
            output = self.mlp_head(pooled.unsqueeze(0))  # [1, num_classes]
            
            batch_outputs.append(output)
            
            # 9. 收集辅助信息
            if return_aux_info:
                if level_info.numel() > 0 and level_info.shape[0] > 1:
                    depths = level_info[1:, 0]  # 排除CLS token
                    unique_levels = depths.unique().tolist()
                    # 安全的token分布计算
                    max_level_for_bincount = min(self.max_level, depths.max().item()) if depths.numel() > 0 else 0
                    token_dist = torch.bincount(depths.long(), minlength=int(max_level_for_bincount + 1)).float()
                    # 填充到完整大小
                    full_token_dist = torch.zeros(self.max_level + 1, device=device)
                    full_token_dist[:token_dist.shape[0]] = token_dist
                else:
                    unique_levels = []
                    full_token_dist = torch.zeros(self.max_level + 1, device=device)
                
                aux_info = {
                    'num_tokens': tokens.shape[0],
                    'levels_used': unique_levels,
                    'token_distribution': full_token_dist
                }
                aux_infos.append(aux_info)
            
            # 10. 特征分析
            if return_features:
                features = self.feature_analyzer(pooled)
                features_list.append(features)
        
        # 11. 合并批次结果
        final_output = torch.cat(batch_outputs, dim=0)  # [B, num_classes]
        
        # 12. 返回结果
        if return_aux_info and return_features:
            return final_output, aux_infos, features_list
        elif return_aux_info:
            return final_output, aux_infos
        elif return_features:
            return final_output, features_list
        else:
            return final_output
    
    def get_tokenizer_loss(self) -> torch.Tensor:
        """获取tokenizer的辅助损失（可学习分割决策的正则化）"""
        if (hasattr(self.fractal_tokenizer, 'split_decision') and 
            self.fractal_tokenizer.split_decision is not None):
            # 计算分割决策网络的正则化损失
            # 鼓励合理的分割概率分布，避免过度分割或分割不足
            
            # 权重正则化
            weight_reg = sum(p.pow(2).sum() for p in self.fractal_tokenizer.split_decision.parameters())
            
            # 层级权重平衡正则化
            level_balance_loss = torch.var(self.level_weights)
            
            total_loss = weight_reg * 0.001 + level_balance_loss * 0.01
            
            return total_loss * self.aux_loss_weight
        
        return torch.tensor(0.0, requires_grad=True, device=self.aux_loss_weight.device)
    
    def analyze_tokenization(self, img: torch.Tensor) -> Dict[str, Any]:
        """分析tokenization过程，返回详细统计信息"""
        with torch.no_grad():
            tokens_list, levels_list = self.fractal_tokenizer(img)
            
            analysis = {
                'batch_size': len(tokens_list),
                'per_image_stats': [],
                'overall_stats': {}
            }
            
            all_levels = []
            total_tokens = 0
            
            for i, (tokens, levels) in enumerate(zip(tokens_list, levels_list)):
                if tokens.numel() == 0:
                    image_stats = {
                        'num_tokens': 0,
                        'levels_used': [],
                        'max_level': 0,
                        'level_distribution': []
                    }
                else:
                    depths = levels[:, 0] if levels.numel() > 0 else torch.tensor([])
                    unique_levels = depths.unique().tolist()
                    max_level = depths.max().item() if depths.numel() > 0 else 0
                    
                    image_stats = {
                        'num_tokens': tokens.shape[0],
                        'levels_used': unique_levels,
                        'max_level': max_level,
                        'level_distribution': torch.bincount(depths.long()).tolist() if depths.numel() > 0 else []
                    }
                    
                    all_levels.extend(unique_levels)
                    total_tokens += tokens.shape[0]
                
                analysis['per_image_stats'].append(image_stats)
            
            # 整体统计
            if all_levels:
                analysis['overall_stats'] = {
                    'total_tokens': total_tokens,
                    'avg_tokens_per_image': total_tokens / len(tokens_list),
                    'unique_levels_used': sorted(list(set(all_levels))),
                    'max_level_overall': max(all_levels),
                    'level_usage_distribution': dict(zip(*torch.unique(torch.tensor(all_levels), return_counts=True)))
                }
            else:
                analysis['overall_stats'] = {
                    'total_tokens': 0,
                    'avg_tokens_per_image': 0,
                    'unique_levels_used': [],
                    'max_level_overall': 0,
                    'level_usage_distribution': {}
                }
            
            return analysis


# 保持向后兼容性的别名
EnhancedFractalViT = NextGenerationFractalViT


# 简化版本 - 向后兼容
class SimpleFractalViT(nn.Module):
    """简化版本，保持向后兼容性，使用合理默认参数"""
    def __init__(self, *, image_size: int, num_classes: int, dim: int = 512, depth: int = 6, heads: int = 8, 
                 mlp_dim: int = 1024, pool: str = 'cls', channels: int = 3, dim_head: int = 64, 
                 dropout: float = 0., emb_dropout: float = 0., min_patch_size: Tuple[int, int] = (4, 4), 
                 max_level: int = 5):
        super().__init__()
        
        # 使用下一代版本作为后端，但使用简化配置
        self.enhanced_model = NextGenerationFractalViT(
            image_size=image_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            min_patch_size=min_patch_size,
            max_level=max_level,
            learnable_split=False,  # 简化版不启用可学习分割
            use_hilbert_encoding=True,
            use_spatial_encoding=True,
            use_feature_enhancement=False,  # 简化版不启用特征增强
            use_dynamic_depth=False
        )
        
        # 保持原有属性以兼容旧代码
        self.fractal_tokenizer = self.enhanced_model.fractal_tokenizer
        self.token_processor = self.enhanced_model.token_processor
        self.pos_embedding = self.enhanced_model.pos_embedding
        self.cls_token = self.enhanced_model.cls_token
        self.dropout = self.enhanced_model.dropout
        self.transformer = self.enhanced_model.transformer
        self.pool = pool
        self.to_latent = self.enhanced_model.to_latent
        self.mlp_head = self.enhanced_model.mlp_head

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.enhanced_model(img, return_attention=False, return_aux_info=False, return_features=False)
