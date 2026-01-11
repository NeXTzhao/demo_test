import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# --- 超参数设置 ---
batch_size = 128        # 批次大小
block_size = 128       # 上下文长度 (原设为10太短了，无法学习长距离依赖，建议至少64或128)
max_iters = 100000       # 训练迭代次数
eval_interval = 500    # 评估间隔
learning_rate = 3e-4   # 学习率 (通常比 1e-4 大一点点效果较好，或者配合 scheduler)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 128          # 嵌入维度
n_head = 4             # 多头注意力的头数
n_layer = 4            # Transformer Block 的层数 (深度)
dropout = 0.2          # Dropout 概率

# --- 数据处理 ---
input_path = 'GPT/input/tanshi.txt'

with open(input_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {s:i for i,s in enumerate(chars)} 
itos = {i:s for i,s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # 确保数据足够长
    if len(data) <= block_size:
        raise ValueError("数据太少，不足以构建一个 block_size。")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- 模型定义 ---

# class Head(nn.Module):
#     """ 单个 Self-Attention Head """
#     def __init__(self, head_size):
#         super().__init__()
#         self.key = nn.Linear(n_embed, head_size, bias=False)
#         self.query = nn.Linear(n_embed, head_size, bias=False)
#         self.value = nn.Linear(n_embed, head_size, bias=False)
#         # tril 用于掩码，确保模型不能看到未来的 token
#         self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         B, T, C = x.shape
#         k = self.key(x)   # (B, T, head_size)
#         q = self.query(x) # (B, T, head_size)
        
#         # 计算注意力分数 (scaled dot-product attention)
#         wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, T)
#         wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
#         wei = F.softmax(wei, dim=-1)
#         wei = self.dropout(wei)
        
#         v = self.value(x) # (B, T, head_size)
#         out = wei @ v     # (B, T, head_size)
#         return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        # Flash Attention 不需要手动 register tril mask，只需要在 forward 时指定 is_causal=True
        self.dropout_val = dropout

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        # 使用 PyTorch 内置的 Flash Attention
        # is_causal=True 会自动处理对角线 mask
        out = F.scaled_dot_product_attention(q, k, v, 
                                             dropout_p=self.dropout_val if self.training else 0, 
                                             is_causal=True)
        return out

class MultiHeadAttention(nn.Module):
    """ 多头注意力机制：并行运行多个 Head """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed) # 投影层，将多头结果融合回 n_embed
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 拼接所有 Head 的输出
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ 前馈神经网络：通常包含一个扩展层 (4x) """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), # 扩展维度
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), # 投影回原维度
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer Block: Communication (Attention) followed by Computation (FFN) """
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # Pre-Norm 结构：先 LayerNorm，再计算，最后残差连接
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding_table = nn.Embedding(block_size, n_embed)
        
        # 堆叠多层 Block
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        
        self.ln_f = nn.LayerNorm(n_embed) # 最终的 LayerNorm
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx 和 targets 都在 device 上
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        
        x = self.blocks(x) # 通过所有 Transformer Blocks
        x = self.ln_f(x)   # 最终归一化
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            return logits, None
        
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx 是 (B, T) 数组
        for _ in range(max_new_tokens):
            # 裁剪 idx 到最新的 block_size 个 token，防止超出位置编码范围
            idx_cond = idx[:, -block_size:]
            
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # 只取最后一个时间步 (B, C)
            
            # --- 采样策略优化 ---
            # 1. Temperature (温度): 
            #    T > 1 会让分布更平缓(更有创造性但易出错)
            #    T < 1 会让分布更尖锐(更保守准确)
            temperature = 1.0 
            logits = logits / temperature
            
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# --- 训练过程 ---

model = GPTLanguageModel()
model = model.to(device)
# 打印参数量
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # 定期评估 loss
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # 采样 batch
    xb, yb = get_batch('train')

    # 训练一步
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --- 生成测试 ---
print("-" * 50)
print("Generating text...")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_ids = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(generated_ids))