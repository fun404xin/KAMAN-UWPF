import torch
import torch.nn as nn
import torch.nn.functional as F

class TSMixer(nn.Module):
    """
    增强版TSMixer，支持两种调用方式：
    1. 列表定义：TSMixer([108, 128, 64, 24])
    2. 直接处理三维输入：输入形状为 [batch, seq_len, feature_dim]
    """
    
    def __init__(self, layer_sizes, seq_len=None, feature_dim=None, num_blocks=3, 
                 dropout=0.1, embedding_size=None):
        """
        参数:
            layer_sizes: 列表 [input_size, hidden1, hidden2, ..., output_size]
                         input_size 应为 seq_len * feature_dim
            seq_len: 时间序列长度（可选，自动推断）
            feature_dim: 特征维度（可选，自动推断）
            num_blocks: TSMixer块的数量，默认3
            dropout: dropout率，默认0.1
            embedding_size: 嵌入维度（可选，默认使用hidden_sizes[0]）
        """
        super(TSMixer, self).__init__()
        
        # 验证输入
        assert len(layer_sizes) >= 2, "layer_sizes必须至少包含输入和输出维度"
        self.layer_sizes = layer_sizes
        self.input_size = layer_sizes[0]
        self.output_size = layer_sizes[-1]
        
        # 自动推断 seq_len 和 feature_dim
        if seq_len is None or feature_dim is None:
            # 尝试推断合理的分解 (例如 18×6=108)
            possible_pairs = []
            for i in range(1, int(self.input_size**0.5) + 1):
                if self.input_size % i == 0:
                    possible_pairs.append((i, self.input_size // i))
                    possible_pairs.append((self.input_size // i, i))
            
            # 选择最合理的组合（优先接近18×6）
            if possible_pairs:
                target_pair = (18, 6)
                possible_pairs.sort(key=lambda x: abs(x[0]-target_pair[0]) + abs(x[1]-target_pair[1]))
                self.seq_len, self.feature_dim = possible_pairs[0]
                print(f"自动推断: seq_len={self.seq_len}, feature_dim={self.feature_dim}")
            else:
                raise ValueError(f"无法分解 input_size={self.input_size} 为 seq_len×feature_dim")
        else:
            self.seq_len = seq_len
            self.feature_dim = feature_dim
        
        # 验证分解是否正确
        assert self.seq_len * self.feature_dim == self.input_size, \
            f"seq_len({self.seq_len})×feature_dim({self.feature_dim}) != input_size({self.input_size})"
        
        # 中间隐藏层大小
        self.hidden_sizes = layer_sizes[1:-1]
        
        # ========== 关键修改：添加Embedding Size控制 ==========
        # 设置embedding_size
        if embedding_size is None:
            # 默认使用第一个隐藏层维度
            self.embedding_size = self.hidden_sizes[0]
        else:
            self.embedding_size = embedding_size
            print(f"设置嵌入维度: embedding_size={self.embedding_size}")
        
        # 1. 输入投影层（现在明确为嵌入层）
        self.embedding = nn.Linear(self.feature_dim, self.embedding_size)
        
        # 2. TSMixer块（使用embedding_size作为特征维度）
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = TSMixerBlock(
                seq_len=self.seq_len,
                feature_dim=self.embedding_size,
                ff_dim=self.embedding_size  # 使用相同的维度
            )
            self.blocks.append(block)
        
        # 3. 特征变换层（从embedding_size变换到其他隐藏层）
        self.feature_layers = nn.ModuleList()
        current_dim = self.embedding_size
        
        for hidden_size in self.hidden_sizes:
            if hidden_size != current_dim:  # 只有当维度变化时才添加层
                self.feature_layers.append(nn.Sequential(
                    nn.Linear(current_dim, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(hidden_size)
                ))
                current_dim = hidden_size
        
        # 4. 输出层（时间维度压缩）
        # 使用当前的最终特征维度
        self.final_feature_dim = current_dim
        self.time_compression = nn.Linear(self.seq_len, self.output_size)
        
        # 5. 可选的RevIN
        self.use_revin = False
        self.revin = None
        
    def set_revin(self, use_revin=True):
        """启用/禁用RevIN标准化"""
        self.use_revin = use_revin
        if use_revin:
            self.revin = RevIN(num_features=self.feature_dim)
    
    def forward(self, x):
        """
        参数:
            x: 可以是以下形状之一:
               - (batch_size, seq_len, feature_dim) [三维]
               - (batch_size, input_size) [二维，自动重塑]
        返回:
            形状为 (batch_size, output_size)
        """
        # 获取批量大小
        batch_size = x.shape[0]
        
        # 自动处理输入形状
        if x.dim() == 2:
            # 二维输入：重塑为三维
            x = x.view(batch_size, self.seq_len, self.feature_dim)
        elif x.dim() == 3:
            # 三维输入：验证形状
            if x.shape[1] != self.seq_len or x.shape[2] != self.feature_dim:
                # 尝试自动调整（如果维度匹配）
                if x.shape[1] * x.shape[2] == self.input_size:
                    # 重塑到正确形状
                    x = x.view(batch_size, self.seq_len, self.feature_dim)
        else:
            raise ValueError(f"输入维度必须为2或3，但得到 {x.dim()}")
        
        # RevIN标准化（如果启用）
        if self.use_revin and self.revin is not None:
            x = self.revin(x, 'norm')
        
        # ========== 关键修改：使用嵌入层 ==========
        # 嵌入层：将特征维度从 feature_dim 投影到 embedding_size
        x = self.embedding(x)  # (batch, seq_len, embedding_size)
        
        # 通过TSMixer块
        for block in self.blocks:
            x = block(x)
        
        # 通过特征变换层
        for layer in self.feature_layers:
            # 在特征维度上变换
            x = layer(x)  # (batch, seq_len, hidden_size)
        
        # 压缩时间维度并输出
        x = x.transpose(1, 2)  # (batch, final_feature_dim, seq_len)
        x = self.time_compression(x)  # (batch, final_feature_dim, output_size)
        x = x.mean(dim=1)  # (batch, output_size) - 平均池化
        
        # RevIN反标准化
        if self.use_revin and self.revin is not None:
            # 需要重塑回三维进行反标准化
            x = x.unsqueeze(-1).transpose(1, 2)  # (batch, output_size, 1) -> (batch, 1, output_size)
            x = self.revin(x, 'denorm')
            x = x.squeeze(1)  # (batch, output_size)
        
        return x

# TSMixerBlock和RevIN类保持不变（无需修改）
class TSMixerBlock(nn.Module):
    """TSMixer核心块"""
    def __init__(self, seq_len, feature_dim, ff_dim, dropout=0.1):
        super(TSMixerBlock, self).__init__()
        
        # 特征混合
        self.feature_mixing = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, feature_dim),
            nn.Dropout(dropout)
        )
        
        # 时间混合
        self.time_mixing = nn.Sequential(
            nn.LayerNorm(seq_len),
            nn.Linear(seq_len, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, seq_len),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
    
    def forward(self, x):
        # 特征混合
        residual = x
        x = self.norm1(x)
        x = self.feature_mixing(x) + residual
        
        # 时间混合
        residual = x
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = self.time_mixing(x) + residual.transpose(1, 2)
        x = x.transpose(1, 2)
        
        return x

class RevIN(nn.Module):
    """可逆实例标准化"""
    def __init__(self, num_features, eps=1e-5):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
    
    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x
    
    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.std(x, dim=dim2reduce, keepdim=True, unbiased=False).detach()
    
    def _normalize(self, x):
        x = x - self.mean
        x = x / (self.stdev + self.eps)
        x = x * self.gamma + self.beta
        return x
    
    def _denormalize(self, x):
        x = x - self.beta
        x = x / (self.gamma + self.eps)
        x = x * (self.stdev + self.eps) + self.mean
        return x