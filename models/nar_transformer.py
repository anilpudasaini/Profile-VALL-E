import torch
import torch.nn as nn

class NARTransformer(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, num_quantizers, vocab_size):
        super(NARTransformer, self).__init__()
        self.phoneme_embedding = nn.Embedding(vocab_size, embed_size)
        self.acoustic_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, embed_size) for _ in range(num_quantizers)]
        )
        self.transformer_layers = nn.ModuleList(
            [nn.Transformer(embed_size, num_heads, num_layers) for _ in range(num_quantizers)]
        )
        self.fc_outs = nn.ModuleList(
            [nn.Linear(embed_size, vocab_size) for _ in range(num_quantizers)]
        )

    def forward(self, phonemes, acoustic_prompt, prev_quantizer_tokens):
        phoneme_embeds = self.phoneme_embedding(phonemes)
        acoustic_prompt_embeds = self.phoneme_embedding(acoustic_prompt)
        outputs = []
        for i in range(len(self.transformer_layers)):
            if i > 0:
                prev_embed = self.acoustic_embeddings[i-1](prev_quantizer_tokens[i-1])
                inputs = phoneme_embeds + prev_embed
            else:
                inputs = phoneme_embeds
            transformer_output = self.transformer_layers[i](inputs, inputs)
            logits = self.fc_outs[i](transformer_output)
            outputs.append(logits)
        return outputs
