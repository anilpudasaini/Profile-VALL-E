import torch
import torch.nn as nn

class PhonemeEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(PhonemeEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, phonemes):
        return self.embedding(phonemes)
