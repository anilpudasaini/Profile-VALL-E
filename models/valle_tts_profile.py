from models.ar_transformer import ARTransformer
from models.nar_transformer import NARTransformer
from transformers import EncodecModel

class VALLE_TTS(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, num_quantizers):
        super(VALLE_TTS, self).__init__()
        self.ar_model = ARTransformer(embed_size, num_heads, num_layers, vocab_size)
        self.nar_model = NARTransformer(embed_size, num_heads, num_layers, num_quantizers, vocab_size)
        self.codec_decoder = EncodecModel.from_pretrained("facebook/encodec_24khz")

    def forward(self, phonemes, acoustic_prompt):
        ar_logits = self.ar_model(phonemes, acoustic_prompt)
        ar_tokens = torch.argmax(ar_logits, dim=-1)
        nar_logits = self.nar_model(phonemes, acoustic_prompt, ar_tokens)
        nar_tokens = [torch.argmax(logit, dim=-1) for logit in nar_logits]
        all_tokens = torch.cat([ar_tokens] + nar_tokens, dim=1)
        waveform = self.codec_decoder.decode(all_tokens)
        return waveform
