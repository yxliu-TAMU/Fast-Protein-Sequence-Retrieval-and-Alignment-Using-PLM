# Levenshtein Transformer for Protein Sequence Alignment (Core Model)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class ProteinLevenshteinTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_encoder_layers=4,
                 num_decoder_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        decoder_layers = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.decoder = TransformerDecoder(decoder_layers, num_decoder_layers)

        self.insertion_head = nn.Linear(d_model, 1)  # Predict number of insertions (regression)
        self.deletion_head = nn.Linear(d_model, 1)   # Predict deletion score (binary classification)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # src: [S, B] (reference sequence)
        # tgt: [T, B] (current hypothesis sequence)

        src_emb = self.pos_encoder(self.embedding(src))  # [S, B, D]
        tgt_emb = self.pos_encoder(self.embedding(tgt))  # [T, B, D]

        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        insertion_logits = self.insertion_head(output).squeeze(-1)  # [T, B]
        deletion_logits = self.deletion_head(output).squeeze(-1)    # [T, B]

        return insertion_logits, deletion_logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
