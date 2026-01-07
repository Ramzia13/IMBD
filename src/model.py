import torch
import torch.nn as nn
import math

class AttentionHead(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.Q = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.K = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])
        self.V = nn.Linear(config["d_embed"], config["head_size"], bias=config["use_bias"])

        self.dropout = nn.Bropout(config["dropout_rate"])

        mask = torch.trial(torch.ones(config["context_size"], config["context_size"]))
        self.register_buffer("mask",mask)

    def forward(self,x);
    B,T,C = x.shape
    Q = self.Q(x)
    K = self.K(x)
    V = self.V(x) 
    
    scores = Q @ K.transpose(1,2)
    scorse = scores / math.sqrt(k.size(-1))

    scores = scores.masked_fill(
        self.mask[:T, :T] == 0, float("-inf")
    )
    weights = torch.softmax(scores, dim=-1)
    weights = self.deopout(wieghts)

    return weights @ V 

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.heads = nn.ModuleList(
            [AttentionHead(config) for _ in range(config["heads_num"])]
        )

        self.proj = nn.Linear(
            config["heads_num"] * config["head_size"],
            config["d_embed"]
        )

        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config["d_embed"], 4 * config["d_embed"]),
            nn.GELU(),
            nn.Linear(4 * config["d_embed"], config["d_embed"]),
            nn.Dropout(config["dropout_rate"])
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn = MultiHeadAttention(config)
        self.ff = FeedForward(config)

        self.ln1 = nn.LayerNorm(config["d_embed"])
        self.ln2 = nn.LayerNorm(config["d_embed"])

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class DemoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_emb = nn.Embedding(
            config["vocabulary_size"], config["d_embed"]
        )

        self.pos_emb = nn.Embedding(
            config["context_size"], config["d_embed"]
        )

        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config["layers_num"])]
        )

        self.ln = nn.LayerNorm(config["d_embed"])

        self.classifier = nn.Linear(
            config["d_embed"], config["num_classes"], bias=False
        )

    def forward(self, token_ids):
        B, T = token_ids.shape

        tok_emb = self.token_emb(token_ids)
        pos_ids = torch.arange(T, device=token_ids.device)
        pos_emb = self.pos_emb(pos_ids)

        x = tok_emb + pos_emb.unsqueeze(0)
        x = self.blocks(x)
        x = self.ln(x)

        x = x.mean(dim=1)   # Mean Pooling
        logits = self.classifier(x)

        return logits
