# 线性层

## 基础概念

**输出分支**：在一个线性层（或类似结构）中，**同一个输入**被同时映射到多个不同的输出空间，每个输出空间就是一个“分支”。每个分支有自己独立的输出维度和参数，但可以通过合并权重矩阵，一次性计算所有分支的输出，提高效率。比如：

1. **QKV 投影**：在 Transformer 的 Attention 里，输入会被同时线性变换为 Q（Query）、K（Key）、V（Value）三组向量，则 Q/K/V 分别是一个分支

2. **MLP 层的 Gate/Up 分支**：有些模型的 MLP 层会把输入同时线性变换为 gate 和 up 两个分支，然后再做激活和融合。

## Linear

### ColumnParallelLinear

```python
class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, 0)
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(output_size, self.tp_size)
                for output_size in self.output_sizes
            ]

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        # 确保加载的权重张量（分片后）和当前参数张量的 shape 完全一致
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
```

**weight_loader**：

- `tp_dim` 表示**在权重参数的哪个维度上进行分片（sharding）**。
- 对于 PyTorch 的 `nn.Linear`，权重 shape 是 `[output_size, input_size]`。
- `ColumnParallelLinear` 是**在输出维度（第 0 维）上分片**，即每张卡负责一部分输出特征，此时`tp_dim=0`

假设：`input_size = 8`，`output_size = 4`，`tp_size = 2`（两卡并行）

- 原始权重的 shape 为：`[4,8]`

- **tp_dim = 0**：在第 0 维（输出维度）分片
  - 每张卡的权重 shape 是 `[2, 8]`
  - 第 1 卡负责前 2 行，第 2 卡负责后 2 行

```shell
 W=torch.range(0,31).reshape(4,8)
 >>> W
tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11., 12., 13., 14., 15.],
        [16., 17., 18., 19., 20., 21., 22., 23.],
        [24., 25., 26., 27., 28., 29., 30., 31.]])
>>> W.shape
torch.Size([4, 8])
>>> W.narrow(0,0,2) # 第一张卡，tp_rank为0，start_idx为0
tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11., 12., 13., 14., 15.]])
>>> W.narrow(0,2,2) # 第二张卡，tp_rank为1，start_idx=1*shard_size=2
tensor([[16., 17., 18., 19., 20., 21., 22., 23.],
        [24., 25., 26., 27., 28., 29., 30., 31.]])
```

**forward**：

在 PyTorch 中，`F.linear(x, weight, bias)` 的计算公式是：`output = X * W^T + b`

- `x` shape: `[batch, input_size]`
- `weight` shape: `[output_size, input_size]`
- `bias` shape: `[output_size]`

则输出 shape 是 `[batch, output_size]`。

### MergedColumnParallelLinear

```python
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias=bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 把 loaded_weight 沿 tp_dim 维度等分成 tp_size 份
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

```

**weight_loader**：

在执行权重加载逻辑之前，权重参数的 shape 已经按照张量并行（TP）方式分割，每张卡只持有自己负责的那部分权重。weight_loader 处理的就是每张卡本地的的分片权重，全局参数和权重在加载时已经被 TP 分割，每张卡只处理自己的分片。

- `loaded_shard_id` 表示当前要加载的是第几个**输出分支**的权重。

  假设有三个输出分支：`output_sizes = [128, 256, 64]`，当`tp_size = 4`（4 卡并行）时，每个分支会在每张卡上分到 `output_size // tp_size` 个输出单元，所以每张卡的本地分支输出维度就是 `[32, 64, 16]`。即在每张卡上，`output_partition_sizes = [32, 64, 16]`。

- `shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size`，计算指定分支在本卡上的起始位置（偏移量）。例如，`loaded_shard_id=1` 时，`shard_offset = (128) // 4 = 32`，表示第二个分支在本卡的起始位置是 32。

- `shard_size = self.output_sizes[loaded_shard_id] // self.tp_size`，计算本分支在本卡上的分片大小。例如，`loaded_shard_id=1` 时，`shard_size = 256 // 4 = 64`。

- `param_data.narrow(self.tp_dim, shard_offset, shard_size)`，取出本卡上属于该分支的参数分片。

- `loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]`，把加载的权重在并行维度（`tp_dim`）上切成 `tp_size` 份，取出本卡的那一份。

### QKVParallelLinear

```python
class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        tp_size = dist.get_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        input_size = self.hidden_size
        # (total_num_heads + 2 * total_num_kv_heads) * head_size
        output_size = (self.num_heads + 2 * self.num_kv_heads) * tp_size * self.head_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,  # q_proj
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj
        ]

        super().__init__(input_size, output_size, bias)

     def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)
```

output_size 是**全局的 QKV 输出总维度**，不是本卡的分片维度。output_sizes：每个分支的输出维度也都乘了 `tp_size`，这样每个分支的 `output_sizes` 也是**全局维度**，不是本地分片维度。

这样做的目的是**让 output_size 和 output_sizes 都是全局维度**，便于和权重加载、切分、拼接等逻辑统一。在权重加载和分片时，再根据 `tp_size` 和 `tp_rank` 计算本卡实际需要的分片（比如 `output_size // tp_size`）。这样可以兼容不同的权重格式（有的 checkpoint 存的是全局权重，有的是分片权重），也方便后续的切分和聚合。

### RowParallelLinear

```python
class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, 1)
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]

        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
```

基本逻辑与 ColumnParallelLinear 一致，这里仅比较两者的不同点：

**分片维度：**

- `ColumnParallelLinear` 在权重的第 0 维（输出维度）分片（`tp_dim=0`）。每张卡负责部分输出特征，权重 shape: `[output_size // tp_size, input_size]`。

- `RowParallelLinear` 在权重的第 1 维（输入维度）分片（`tp_dim=1`）。每张卡负责部分输入特征，权重 shape: `[output_size, input_size // tp_size]`。

**输入输出 shape**：

- `ColumnParallelLinear`：输入 `x` shape: `[batch, input_size]`（不变），输出 shape: `[batch, output_size // tp_size]`（每卡只输出一部分）
- `RowParallelLinear`：输入 `x` shape: `[batch, input_size // tp_size]`（每卡只处理一部分输入），输出 shape: `[batch, output_size]`（每卡都输出完整的 output_size）

**前向传播**：

- `ColumnParallelLinear`：每张卡计算 `y_partial = x @ W_partial.T`，输出 shape: `[batch, output_size // tp_size]`。每张卡负责一部分输出特征，**所有卡的输出拼起来就是完整输出**，不需要 all_reduce 聚合，只需后续 gather 或拼接即可
- `RowParallelLinear`：每张卡计算 `y_partial = x_partial @ W_partial.T`，输出 shape: `[batch, output_size]`，但每张卡只算了自己负责的输入部分，**输出只是部分和**，不是最终结果。所以需要 `all_reduce` 把所有卡的 `y_partial` 相加，得到完整的 `[batch, output_size]`

**典型应用场景**：

- `ColumnParallelLinear`：QKV 合并线性层、MLP 的输出层等。
- `RowParallelLinear`：Embedding 后的第一层线性、MLP 的输入层等。
  - Embedding 层输出 shape 通常是 `[B, S, H]`，`hidden_size` 很大（比如 4096、8192），用 RowParallelLinear，可以把输入特征（hidden_size）在多卡上分片，每张卡只处理一部分输入特征，显存占用大大降低。
  - 多头注意力机制中，每个 head 会输出一个 `[B, S, head_dim]` 的张量，所有 head 的输出会在最后一个维度（特征维度）上拼接（concat），得到 `[B, S, num_heads * head_dim]`，也就是 `[B, S, H]`。这个拼接后的张量会经过一个线性变换（ **o_proj**），由于拼接后的 shape 中，hidden_size 很大，因此`o_proj` 定义为 RowParallelLinear，最后通过 all_reduce 聚合，得到完整的 `[B, S, H]` 张量，作为 **MLP 层的输入**。

| 类别                 | 分片维度 | 权重 shape                    | 输入 shape              | 输出 shape               | 前向聚合方式 |
| -------------------- | -------- | ----------------------------- | ----------------------- | ------------------------ | ------------ |
| ColumnParallelLinear | 0        | [output_size//tp, input_size] | [batch, input_size]     | [batch, output_size//tp] | 拼接         |
| RowParallelLinear    | 1        | [output_size, input_size//tp] | [batch, input_size//tp] | [batch, output_size]     | all_reduce   |

## MLP

以典型的 Transformer Block 为例，MLP 结构：输入 → 线性变换（升维）→ 激活 → 线性变换（降维）→ 输出 MLP

1. **输入**：多头注意力的输出，shape 通常为 `[B, S, H]`。
2. **第一层线性变换**（输入层，通常用 **ColumnParallelLinear**）：
   - 公式：`hidden1 = x @ W1^T + b1`
   - 输入 shape: `[B, S, H]`
   - 权重 W1 shape: `[I/4, H]`
   - 输出 shape: `[B, S, I/4]`
3. **激活函数**（如 GELU、Silu 等）：
   - 公式：`hidden2 = activation(hidden1)`
   - 输出 shape: `[B, S, I/4]`
4. **第二层线性变换**（输出层，通常用 **RowParallelLinear**）：
   - 公式：`output = hidden2 @ W2^T + b2`
   - 权重 W2 shape: `[H, I/4]`
   - 输出 shape: `[B, S, H]`

由于典型的 Transformer Block 中，MLP 结构中有两个线性层，因此两个线性层有四种这方式：

| 组合  | 第一层                   | 第二层                |
| ----- | ------------------------ | --------------------- |
| A     | RowParallelLinear        | ColumnParallelLinear  |
| B     | RowParallelLinear        | RowParallelLinear     |
| C     | ColumnParallelLinear     | ColumnParallelLinear  |
| **D** | **ColumnParallelLinear** | **RowParallelLinear** |

假设：H = 4096，I = 11008，tp = 4

| 方案      | 第一层输入           | 第一层权重           | 第一层输出                      | 第二层输入           | 第二层权重           | 第二层输出                              | 激活显存（第二层输入+输出） | 第一层输出到第二层输入 | 通信量                                                     |
| --------- | -------------------- | -------------------- | ------------------------------- | -------------------- | -------------------- | --------------------------------------- | --------------------------- | ---------------------- | ---------------------------------------------------------- |
| A Row,Col | [B,S,H/4] [B,S,1024] | [I,H/4] [11008,1024] | [B,S,11008] （all-reduce 聚合） | [B,S,I] [B,S,11008]  | [H/4,I] [1024,11008] | [B,S,1024] （最终 gather 为[B,S,4096]） | 11008+1024=12032            | all-reduce             | 第一层 all-reduce [B,S,11008] 第二层 gather [B,S,4096]     |
| B Row,Row | [B,S,H/4] [B,S,1024] | [I,H/4] [11008,1024] | [B,S,11008] （all-reduce 聚合） | [B,S,I/4] [B,S,2752] | [H,I/4] [4096,2752]  | [B,S,4096] （all-reduce 聚合）          | 2752+4096=6848              | all-reduce             | 第一层 all-reduce [B,S,11008] 第二层 all-reduce [B,S,4096] |
| C Col,Col | [B,S,H] [B,S,4096]   | [I/4,H] [2752,4096]  | [B,S,I/4] [B,S,2752]            | [B,S,I/4] [B,S,2752] | [H/4,I] [1024,11008] | 权重 shape 不匹配，无法直接操作         |                             |                        |                                                            |
| D Col,Row | [B,S,H] [B,S,4096]   | [I/4,H] [2752,4096]  | [B,S,I/4] [B,S,2752]            | [B,S,I/4] [B,S,2752] | [H,I/4] [4096,2752]  | [B,S,4096] （all-reduce 聚合）          | 2752+4096=6848              | 无需操作               | 第二层 all-reduce [B,S,4096]                               |

目前**主流大模型训练和推理框架都采用 D 方案，是工程和通信权衡的最佳结果。**

# 位置编码

## 方程

RoPE 的方程式：

$$
PE(pos,2i)&=sin(\frac{pos}{10000^{2i/d}}) \\
PE(pos,2i+1)&=cos(\frac{pos}{10000^{2i/d}})\\
$$

其中：

- `pos` 是指 token 在输入序列中的**位置索引**。对于每一个输入 token，可以通过其在序列中的索引来标识其具体位置。在一个长度为 `L` 的输入序列中，位置索引 `pos` 取值范围从 `0` 到 `L-1`。
- `i` 是指位置编码向量中的**分量索引**（维度索引）。每个位置的编码向量有多个分量，对应于模型嵌入的维度。在一个维度为 `d` 的嵌入空间中，每个位置 `pos` 需要一个维度为 `d` 的位置编码向量，因此有 `d` 个分量，则 `i` 值可能为 `0` 到 `d/2-1`。
- `d` 是模型维度，即`hidden_size`。
- 10,000 是**基本波长**（下文简称为 $\theta$），我们根据分量索引对其进行拉伸或压缩。

## 代码

### init

```python
class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        assert rotary_dim == head_size
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base**(torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        # 外积，类似于torch.outer(i,j)
        # 得到矩阵大小为 max_position_embeddings * (rotary_dim/2)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # cache矩阵大小为 max_position_embeddings * rotary_dim
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)
```

参考上面的公式，这里的对应关系：

| 公式参数 | 代码变量                            | 说明                                     |
| -------- | ----------------------------------- | ---------------------------------------- |
| pos      | max_position_embeddings             | 模型支持的最大序列长度                   |
| d        | self.rotary_dim                     | rotary 作用的维度（通常等于 head_size）  |
| i        | torch.arange(0, self.rotary_dim, 2) | 分量索引（偶数分量， RoPE 通常两两一组） |
| $\theta$ | self.base                           | 波长基数                                 |

### Forward

```python

class RotaryEmbedding(nn.Module):
    # ...
    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions.flatten() # 实际输入的位置
        num_tokens = positions.shape[0] # 实际输入的序列长度
        # cos_sin_cache矩阵: [max_position_embeddings , rotary_dim]
        # cos_sin的矩阵维度为：[num_tokens, rotary_dim]
        cos_sin = self.cos_sin_cache[positions]
        # cos 矩阵维度：[num_tokens, rotary_dim/2]
        cos, sin = cos_sin.chunk(2, dim=-1) # 分为两块
        query_shape = query.shape
        # query 张量维度：[num_tokens, num_heads, head_size]
        query = query.view(num_tokens, -1, self.head_size) #三维张量
        query = apply_rotary_emb(query, cos, sin).view(query_shape)
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        return query, key

def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # 向张量中添加维度,在指定位置插入一个大小为 1 的新维度
    # 得到一个3维张量[num_tokens, 1, rotary_dim/2]
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    # 三维张量 [num_tokens, num_heads, head_size/2]
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
    # 矩阵乘法，应用分块旋转矩阵
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)

```

在以上代码中，假设：

- `num_tokens = 4`，`num_heads = 2`，`head_size = 8`，`rotary_dim = 8`
- `max_position_embeddings = 32768`

输入张量：

```python
query.shape = (num_tokens, num_heads, head_size) # (4, 2, 8)
```

取出实际位置的 cos/sin：

```python
# num_tokens=4, 假设 positions = [0, 1, 2, 3]

cos_sin_cache.shape = (max_position_embeddings, rotary_dim) # (32768, 8)
# cos_sin的矩阵维度为：[num_tokens, rotary_dim]
cos_sin = self.cos_sin_cache[positions] # (4,8)
# cos 矩阵维度：[num_tokens, rotary_dim/2]
cos, sin = cos_sin.chunk(2, dim=-1) # (4,4)
```

在 apply_rotary_emb 里：

```python
x = query  # (4, 2, 8)
cos = cos.unsqueeze(-2)  # (4, 1, 4)
sin = sin.unsqueeze(-2)  # (4, 1, 4)
x1, x2 = torch.chunk(x, 2, dim=-1)  # x1, x2: (4, 2, 4)
```

**PyTorch 广播规则：**

- 维度从右往左对齐，不一致的维度如果有 1 会自动扩展。
- 所以 `(4, 2, 4)` 和 `(4, 1, 4)` 可以广播成 `(4, 2, 4)`，即每个 head 都用同一个 cos/sin.

### Cache

```python
@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb

```

在同一个模型中，所有的 attention 层使用相同的旋转位置编码配置，`get_rope` 的参数通常是固定的：因此只需要一个 `RotaryEmbedding` 实例。

在 `get_rope` 函数中，`@lru_cache(1)` 缓存的是 `RotaryEmbedding` 实例。这个实例的主要内存占用来自于 `cos_sin_cache` 这个缓冲区，其最终形状是 `[max_position_embeddings, rotary_dim]`。因此，`lru_cache` 中缓存的 `RotaryEmbedding` 实例的主要内存占用是：

**内存大小 = `max_position_embeddings * rotary_dim * 4` 字节**

其中：

- `max_position_embeddings` 是最大位置编码数量
- `rotary_dim` 是旋转维度（等于 `head_dim`）
- `4` 是因为使用 `torch.float` (float32) 类型，每个元素占 4 字节

以[通义千问 3-8B](https://www.modelscope.cn/models/Qwen/Qwen3-8B/file/view/master/config.json?status=1)为例，内存占用约为：

$40960 * 4096 * 4 /1024 /1024=640\;MB$

# embed_head

## VocabParallelEmbedding

```python
class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,  # vocab size
        embedding_dim: int,  # hidden size
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        # 每张卡负责一部分单词的嵌入
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.embedding_dim = embedding_dim
        # 每张卡加载 embed 矩阵的一个分片
        self.weight = nn.Parameter(torch.empty(
            self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        # x 是用户输入的Token ids
        if self.tp_size > 1:
            # 检查输入的Token ID 是否属于当前卡负责的分片范围
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 将属于当前分片的Token ID转化为本地索引
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(-1) * y
            dist.all_reduce(y)
        return y

```

重点解释一下这里的 Forward，如果是单卡则不需要进行分批，直接嵌入查找并返回结果即可。

当张量并行时，由于每张卡上只负责加载一部分的嵌入权重，每张卡只负责自己分片内的词汇嵌入。当用户输入 TokenIDs 后，需要检查哪些 ID 属于当前分片，`vocab_start_idx` 和 `vocab_end_idx` 定义了当前 GPU 负责的词汇 ID 范围。

由于`F.embedding` 会使用本地的权重分片进行嵌入查找，因此还需要将 Token ID 转化为本地的索引（减去起始偏移）。

举例说明：

```shell
# 配置
vocab_size = 12 # 词汇表大小
tp_size = 4 # 4个GPU
embedding_dim = 8

# 每个 GPU 的分片配置
GPU 0: vocab_range = [0, 3)   # 负责 token ID: 0,1,2
GPU 1: vocab_range = [3, 6)   # 负责 token ID: 3,4,5
GPU 2: vocab_range = [6, 9)   # 负责 token ID: 6,7,8
GPU 3: vocab_range = [9, 12)  # 负责 token ID: 9,10,11

# 输入示例
input_ids = [1, 4, 7, 10]  # 4 个 token
```

以 GPU1 为例，其执行过程如下：

```shell
# GPU 1: vocab_start_idx=3, vocab_end_idx=6
x = torch.tensor([1, 4, 7, 10])

# 1. 创建掩码
mask = (x >= 3) & (x < 6)  # [False, True, False, False]

# 2. 调整 token ID
x = mask * (x - 3)  # [0, 1, 0, 0]  (只有 token 4 属于当前分片，转为本地索引 1)

# 3. 嵌入查找
y = F.embedding(x, weight)  # weight.shape = [3, 8]
# 结果: [[emb_0], [emb_1], [emb_0], [emb_0]]  (emb_1 是正确的，其他是无效的)
# y.shape = [4,8]

# 4. 应用掩码
# mask shape:[4] --> [4,1]
mask = mask.unsqueeze(-1)  # [False, True, False, False] -> [[False], [True], [False], [False]]
y = mask * y  # 只保留第二个位置的结果，其他置零
# 结果: [[0,0,0,0,0,0,0,0], [正确的emb_1], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]]

# 5. 跨 GPU 聚合
dist.all_reduce(y)  # 所有 GPU 的结果相加，最终每个位置只有一个 GPU 贡献非零值
```

## ParallelLMHead

```python
class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__(num_embeddings, embedding_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty(
                self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)


    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            # Cumulative sequence lengths for Q
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        #
        logits = F.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(
                self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits

```

同样，重点看一下 Forward 部分。

1. 在代码中，对 prefill 阶段做了特殊处理。 **prefill 阶段**，模型需要处理完整的输入序列来构建 KV Cache。但对于语言模型，只需要预测每个序列的**最后一个 token**。在 decode 阶段，每次输入只有一个 Token，因此不需要提取。

- `cu_seqlens_q` （Cumulative sequence lengths for Q）是 Q 的累积序列长度数组。

```python
# 假设有3个序列，长度分别为 [5, 7, 8]
cu_seqlens_q = [0, 5, 12, 20]  # 累积长度
last_indices = context.cu_seqlens_q[1:] - 1 = [4, 11, 19]  # 每个序列的最后一个token位置
# 原始 x 长度为20，则 x 的形状: [20, hidden_dim] (所有token的hidden states)
x = x[last_indices].contiguous()
# 处理后 x 形状: [3, hidden_dim] (只保留每个序列的最后一个token)
```

2. 计算 logits：执行线性变换`logits = x @ weight.T + bias`，将隐藏状态映射到词汇表空间。在张量并行下，每个 GPU 只计算词汇表的一部分
3. 聚合 logits：`gather` 将所有 GPU 的 logits 收集到 rank 0，然后拼接成完整的词汇表。只有 rank 0 的 GPU 得到完整结果，其他 GPU 返回 `None`

举例说明：

```shell
# 假设有2个GPU进行TP并行，配置如下
vocab_size = 10000
tp_size = 2
hidden_dim = 512

# GPU 分片
GPU 0: 负责 vocab [0:5000]
GPU 1: 负责 vocab [5000:10000]

# 输入3个序列
x.shape = [3, 512]  # 3个序列的最后token的hidden states

# 在每个 GPU 上
logits = F.linear(x, weight, bias)
# GPU 0: logits.shape = [3, 5000] (词汇表前半部分的logits)
# GPU 1: logits.shape = [3, 5000] (词汇表后半部分的logits)

# gather 操作，将所有GPU的logits收集到GPU 0
dist.gather(logits, all_logits, 0)

# 在 GPU 0 上拼接
if self.tp_rank == 0:
    logits = torch.cat([gpu0_logits, gpu1_logits], dim=-1)
    # 最终结果: logits.shape = [3, 10000] (完整词汇表的logits)
else:
    logits = None  # 其他GPU返回None
```

# 模型加载

```python
def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    //
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys(): # safetensor的原始权重名称
                for k in packed_modules_mapping:
                  #  例如 q_proj in "model.layers.0.self_attn.q_proj.weight"
                    if k in weight_name:
                        # "qkv_proj", "q"
                        v, shard_id = packed_modules_mapping[k]
                        # "model.layers.0.self_attn.q_proj.weight" --> "model.layers.0.self_attn.qkv_proj.weight", weight_name不变，parma_name是新的
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))

```

在 `Qwen3ForCausalLM` 定义了类的属性 `packed_modules_mapping`，映射的核心作用是**权重名称转换和分片指导**。

```python
class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
```

在原始的 Qwen3 模型中，safetensor 文件里存储的权重名称是分离的：

```json
{
  "metadata": {
    "total_size": 65524246528
  },
  "weight_map": {
    "lm_head.weight": "model-00017-of-00017.safetensors",
    "model.embed_tokens.weight": "model-00001-of-00017.safetensors",
    "model.layers.0.input_layernorm.weight": "model-00001-of-00017.safetensors",
    "model.layers.0.mlp.down_proj.weight": "model-00001-of-00017.safetensors",
    "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00017.safetensors",
    "model.layers.0.mlp.up_proj.weight": "model-00001-of-00017.safetensors",
    "model.layers.0.post_attention_layernorm.weight": "model-00001-of-00017.safetensors",
    "model.layers.0.self_attn.k_norm.weight": "model-00001-of-00017.safetensors",
    "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00017.safetensors",
    "model.layers.0.self_attn.o_proj.weight": "model-00001-of-00017.safetensors",
    "model.layers.0.self_attn.q_norm.weight": "model-00001-of-00017.safetensors",
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00017.safetensors",
    "model.layers.0.self_attn.v_proj.weight": "model-00001-of-00017.safetensors",
  }

layers.0.self_attn.q_proj.weight    # Query 投影权重
layers.0.self_attn.k_proj.weight    # Key 投影权重
layers.0.self_attn.v_proj.weight    # Value 投影权重
layers.0.mlp.gate_proj.weight       # Gate 投影权重
layers.0.mlp.up_proj.weight         # Up 投影权重
```

但在 nano-vllm 的实现中，为了效率，这些权重被**融合**成了单一模块：

```python
# 在 Qwen3Attention 中
self.qkv_proj = QKVParallelLinear(...)  # 融合了 q_proj + k_proj + v_proj

# 在 Qwen3MLP 中
self.gate_up_proj = MergedColumnParallelLinear(...)  # 融合了 gate_proj + up_proj
```

此时，模型的结构发生了变化：

![image-20250701153036406](/Users/qcy/Library/Application Support/typora-user-images/image-20250701153036406.png)

因此，在加载模型时，需要通过权重名称匹配进行转化，将原来分离的权重合并为当前实现支持的融合权重。

```python
def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    for weight_name in f.keys():  # 来自 safetensor 文件的原始权重名称
        for k in packed_modules_mapping:
          # 例如 "q_proj" in "layers.0.self_attn.q_proj.weight"
            if k in weight_name:
                v, shard_id = packed_modules_mapping[k]  # ("qkv_proj", "q")
                param_name = weight_name.replace(k, v)   # 转换成融合模块的名称
                # "layers.0.self_attn.q_proj.weight" → "layers.0.self_attn.qkv_proj.weight"
                # 得到nano-vllm实现中的参数，Parameter对象，data大小为qkv融合后的size
                param = model.get_parameter(param_name)
```

```shell
# 原始 safetensor 中的权重名称 → 转换后的参数名称
"layers.0.self_attn.q_proj.weight" → "layers.0.self_attn.qkv_proj.weight" (shard_id="q")
"layers.0.self_attn.k_proj.weight" → "layers.0.self_attn.qkv_proj.weight" (shard_id="k")
"layers.0.self_attn.v_proj.weight" → "layers.0.self_attn.qkv_proj.weight" (shard_id="v")
"layers.0.mlp.gate_proj.weight"    → "layers.0.mlp.gate_up_proj.weight"   (shard_id=0)
"layers.0.mlp.up_proj.weight"      → "layers.0.mlp.gate_up_proj.weight"   (shard_id=1)
```

融合模块通过自定义的 weight_loader 来处理权重加载的逻辑，以 QKV 为例：

```python
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + \
                self.num_kv_heads * self.head_size
        # 获取待赋值部分Tensor
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 将已加载的模型权重，如"layers.0.self_attn.q_proj.weight"的部分进行分块
        loaded_weight = loaded_weight.chunk(
            self.tp_size, self.tp_dim)[self.tp_rank]
        assert param_data.size() == loaded_weight.size()
        # 将分块后的权重赋值到param_data
        param_data.copy_(loaded_weight)

```

# ModelRunner

## KV_Cache

在 Multi-Head Attention 中，每个 head 都有自己独立的 K 和 V；每个 token 在每个 attention head 中都会产生一个 K 向量和一个 V 向量。因此，**每个 token** 在单层中的 KV cache 存储需求：

```python
# 对于一个 token，在一层中需要存储：
k_storage_per_token = num_kv_heads * head_dim * dtype_size  # K 向量
v_storage_per_token = num_kv_heads * head_dim * dtype_size  # V 向量
total_per_token_per_layer = 2 * num_kv_heads * head_dim * dtype_size
```

如果一个 Block 中包含多个 tokens，则单层中一个 Block 需要的存储空间为：

```python
block_storage_per_layer = block_size * 2 * num_kv_heads * head_dim * dtype_size
```

因此，整个 Transformer 模型中，一个 Block 的存储空间为：

```python
block_bytes = 2 * num_hidden_layers * block_size * num_kv_heads * head_dim * dtype_size
```
