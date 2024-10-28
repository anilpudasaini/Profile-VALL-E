import torch
import torch.nn as nn

class ARTransformer(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, vocab_size):
        super(ARTransformer, self).__init__()
        self.phoneme_embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.Transformer(embed_size, num_heads, num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, phonemes, acoustic_prompt):
        phoneme_embeds = self.phoneme_embedding(phonemes)
        acoustic_prompt_embeds = self.phoneme_embedding(acoustic_prompt)
        inputs = torch.cat((acoustic_prompt_embeds, phoneme_embeds), dim=1)
        outputs = self.transformer(inputs, inputs)
        logits = self.fc_out(outputs)
        return logits
