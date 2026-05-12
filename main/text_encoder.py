"""
文本编码器模块
============

将音乐文本描述转换为嵌入向量,用于条件生成。
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional


class SimpleTextEncoder(nn.Module):
    """
    简单的文本编码器,基于词嵌入和 Transformer。
    
    适用于音乐特征文本,如: "piano happy melody in C"
    """
    
    def __init__(self, 
                 vocab_size: int = 1000,
                 embed_dim: int = 768,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 max_seq_len: int = 32,
                 dropout: float = 0.1):
        """
        Args:
            vocab_size: 词汇表大小
            embed_dim: 嵌入维度 (应与 DiT 的 embed_dim 一致)
            num_layers: Transformer 层数
            num_heads: 注意力头数
            max_seq_len: 最大文本长度
            dropout: Dropout 率
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # 词嵌入
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出投影 (提取 [CLS] token 的表示)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
    def forward(self, 
                token_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            token_ids: 词 ID 序列 (B, L)
            attention_mask: 注意力掩码 (B, L), 1 表示有效, 0 表示填充
            
        Returns:
            文本嵌入 (B, embed_dim)
        """
        B, L = token_ids.shape
        
        # 词嵌入 + 位置编码
        x = self.token_embed(token_ids)  # (B, L, D)
        x = x + self.pos_embed[:, :L, :]  # (B, L, D)
        
        # 准备 Transformer 的掩码 (True 表示忽略)
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None
            
        # Transformer 编码
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # (B, L, D)
        
        # 提取 [CLS] token (第一个 token) 的表示
        cls_emb = x[:, 0, :]  # (B, D)
        
        # 输出投影
        text_emb = self.output_proj(cls_emb)  # (B, D)
        
        return text_emb


class MusicTextVocabulary:
    """
    音乐文本词汇表
    
    用于将音乐描述文本转换为 token IDs。
    """
    
    def __init__(self):
        """初始化词汇表"""
        
        # 特殊 token
        self.pad_token = "<PAD>"
        self.cls_token = "<CLS>"
        self.unk_token = "<UNK>"
        
        # 乐器
        instruments = [
            "piano", "guitar", "bass", "drums", "violin", "cello",
            "flute", "clarinet", "saxophone", "trumpet", "trombone",
            "organ", "synth", "strings"
        ]
        
        # 情绪
        moods = ["calm", "happy", "sad", "energetic"]
        
        # 强度
        intensities = ["gentle", "expressive", "powerful"]
        
        # 调性
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        
        # 速度
        tempos = ["slow", "moderate", "fast", "very fast", "adagio", "andante", 
                  "allegro", "presto"]
        
        # 其他描述词
        descriptors = [
            "melody", "with", "rich", "harmonies", "in", "and",
            "major", "minor", "scale", "arpeggio", "chord", "progression"
        ]
        
        # 构建词汇表
        self.vocab = [self.pad_token, self.cls_token, self.unk_token]
        self.vocab.extend(instruments)
        self.vocab.extend(moods)
        self.vocab.extend(intensities)
        self.vocab.extend(keys)
        self.vocab.extend(tempos)
        self.vocab.extend(descriptors)
        
        # 创建索引映射
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        
        # 特殊 token ID
        self.pad_id = self.token_to_id[self.pad_token]
        self.cls_id = self.token_to_id[self.cls_token]
        self.unk_id = self.token_to_id[self.unk_token]
        
    def __len__(self) -> int:
        """返回词汇表大小"""
        return len(self.vocab)
        
    def encode(self, text: str, max_length: int = 32) -> Dict[str, torch.Tensor]:
        """
        编码文本为 token IDs
        
        Args:
            text: 文本描述,如 "piano happy melody in C"
            max_length: 最大长度
            
        Returns:
            字典包含:
                - token_ids: Token ID 张量 (L,)
                - attention_mask: 注意力掩码 (L,)
        """
        # 分词 (简单的空格分割)
        tokens = text.lower().split()
        
        # 添加 [CLS] token
        tokens = [self.cls_token] + tokens
        
        # 转换为 ID
        token_ids = []
        for token in tokens:
            token_id = self.token_to_id.get(token, self.unk_id)
            token_ids.append(token_id)
            
        # 截断或填充
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            attention_mask = [1] * max_length
        else:
            attention_mask = [1] * len(token_ids) + [0] * (max_length - len(token_ids))
            token_ids = token_ids + [self.pad_id] * (max_length - len(token_ids))
            
        return {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
        
    def encode_batch(self, texts: List[str], max_length: int = 32) -> Dict[str, torch.Tensor]:
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            max_length: 最大长度
            
        Returns:
            字典包含:
                - token_ids: Token ID 张量 (B, L)
                - attention_mask: 注意力掩码 (B, L)
        """
        batch_token_ids = []
        batch_attention_mask = []
        
        for text in texts:
            encoded = self.encode(text, max_length)
            batch_token_ids.append(encoded['token_ids'])
            batch_attention_mask.append(encoded['attention_mask'])
            
        return {
            'token_ids': torch.stack(batch_token_ids),
            'attention_mask': torch.stack(batch_attention_mask)
        }
        
    def decode(self, token_ids: torch.Tensor) -> str:
        """
        解码 token IDs 为文本
        
        Args:
            token_ids: Token ID 张量 (L,)
            
        Returns:
            文本字符串
        """
        tokens = []
        for token_id in token_ids.tolist():
            if token_id == self.pad_id:
                break
            token = self.id_to_token.get(token_id, self.unk_token)
            if token not in [self.cls_token, self.pad_token]:
                tokens.append(token)
                
        return ' '.join(tokens)


def create_text_encoder(
    vocab_size: int = 100,
    embed_dim: int = 768,
    num_layers: int = 4,
    num_heads: int = 8,
    max_seq_len: int = 32,
    dropout: float = 0.1
) -> SimpleTextEncoder:
    """
    创建文本编码器的工厂函数
    
    Args:
        vocab_size: 词汇表大小
        embed_dim: 嵌入维度
        num_layers: Transformer 层数
        num_heads: 注意力头数
        max_seq_len: 最大文本长度
        dropout: Dropout 率
        
    Returns:
        文本编码器模型
    """
    return SimpleTextEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        dropout=dropout
    )


# 创建全局词汇表实例
MUSIC_VOCAB = MusicTextVocabulary()


if __name__ == "__main__":
    """测试文本编码器"""
    
    print("=== 测试音乐文本编码器 ===\n")
    
    # 创建词汇表
    vocab = MusicTextVocabulary()
    print(f"词汇表大小: {len(vocab)}")
    print(f"词汇表前 20 个词: {vocab.vocab[:20]}\n")
    
    # 测试编码
    test_texts = [
        "piano happy melody in C",
        "guitar and violin gentle melody in G",
        "drums energetic in A"
    ]
    
    print("测试文本编码:")
    for text in test_texts:
        encoded = vocab.encode(text, max_length=16)
        print(f"\n原文: {text}")
        print(f"Token IDs: {encoded['token_ids'][:10]}...")
        print(f"Attention Mask: {encoded['attention_mask'][:10]}...")
        decoded = vocab.decode(encoded['token_ids'])
        print(f"解码: {decoded}")
        
    # 测试批量编码
    print("\n\n测试批量编码:")
    batch_encoded = vocab.encode_batch(test_texts, max_length=16)
    print(f"Batch Token IDs shape: {batch_encoded['token_ids'].shape}")
    print(f"Batch Attention Mask shape: {batch_encoded['attention_mask'].shape}")
    
    # 测试编码器模型
    print("\n\n测试编码器模型:")
    encoder = create_text_encoder(
        vocab_size=len(vocab),
        embed_dim=768,
        num_layers=4,
        num_heads=8,
        max_seq_len=32
    )
    
    print(f"模型参数量: {sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M")
    
    # 前向传播
    with torch.no_grad():
        text_emb = encoder(
            batch_encoded['token_ids'],
            batch_encoded['attention_mask']
        )
    print(f"文本嵌入 shape: {text_emb.shape}")
    print(f"文本嵌入范围: [{text_emb.min():.3f}, {text_emb.max():.3f}]")
    
    print("\n✓ 文本编码器测试通过!")
