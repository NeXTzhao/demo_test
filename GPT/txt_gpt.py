import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import time

# ==========================================
# 1. 超参数设置 (针对小说数据进行了调整)
# ==========================================
batch_size = 64        # 批次大小
block_size = 512       # 上下文长度 (小说需要更长的上文，建议 256 或 512)
max_iters = 20000       # 训练迭代次数
eval_interval = 500    # 每隔多少步评估一次
learning_rate = 3e-4   # 学习率
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 256          # 嵌入维度 (调大一点，为了捕捉更丰富的语义)
n_head = 8             # 多头注意力的头数
n_layer = 6            # 层数 (加深网络)
dropout = 0.2          # 防止死记硬背

print(f"Using device: {device}")

# ==========================================
# 2. 数据加载与预处理 (支持读取多个文件)
# ==========================================

# !!! 请将此处修改为你存放 txt 文件的文件夹路径 !!!
data_dir = 'GPT/data' 

# 如果文件夹不存在，创建一个假的（防止报错）
if not os.path.exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, 'dummy.txt'), 'w', encoding='utf-8') as f:
        f.write("飞雪连天射白鹿，笑书神侠倚碧鸳。" * 1000)
    print(f"注意：未找到数据目录，已创建 {data_dir} 并生成测试数据。")

# 读取目录下所有 txt 文件
txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
raw_text = ""

print(f"正在读取 {len(txt_files)} 个文件...")
for file_path in txt_files:
    try:
        # 尝试 UTF-8 读取
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text += f.read() + "\n"
    except UnicodeDecodeError:
        # 如果 UTF-8 失败，尝试 GBK (很多老小说txt是GBK编码)
        try:
            with open(file_path, 'r', encoding='gb18030') as f:
                raw_text += f.read() + "\n"
        except:
            print(f"Skipping {file_path}: 编码错误")

print(f"数据集总长度: {len(raw_text)} 字符")

# 构建词表
chars = sorted(list(set(raw_text)))
vocab_size = len(chars)
print(f"词表大小 (Vocab size): {vocab_size}")

stoi = {s:i for i,s in enumerate(chars)} 
itos = {i:s for i,s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi] # 忽略未知字符
decode = lambda l: ''.join([itos[i] for i in l])

# 划分训练集和验证集
data = torch.tensor(encode(raw_text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
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

# ==========================================
# 3. 优化后的模型架构 (Flash Attention + GELU)
# ==========================================

class Head(nn.Module):
    """ 使用 Flash Attention 的 Head """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout_val = dropout

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        # 使用 PyTorch 内置的 Flash Attention (速度更快，显存更省)
        out = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout_val if self.training else 0, 
            is_causal=True
        )
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(), # 使用 GELU 代替 ReLU，效果更好
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

        # 初始化权重 (有助于收敛)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            
            # 温度采样
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==========================================
# 4. 训练循环
# ==========================================

model = GPTLanguageModel()
model = model.to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("开始训练...")
start_time = time.time()

for iter in range(max_iters):
    
    # 评估
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # 保存模型检查点 (可选)
        if iter > 0:
            torch.save(model.state_dict(), 'wuxia_gpt_checkpoint.pth')

    # 采样与反向传播
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # 梯度裁剪 (防止梯度爆炸)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()

end_time = time.time()
print(f"训练结束，耗时: {(end_time - start_time)/60:.2f} 分钟")

# ==========================================
# 5. 生成演示
# ==========================================
print("\n" + "="*30)
print("模型生成演示:")
print("="*30)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
# 让模型多生成一点 (1000字)
generated_ids = model.generate(context, max_new_tokens=1000, temperature=0.8)[0].tolist()
print(decode(generated_ids))