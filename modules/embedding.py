# Copyright    2023                             (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


from typing import List, Union
from transformers import AutoTokenizer, AutoModel

class ProfileEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.pooler = nn.Sequential(
            nn.Linear(2 * self.text_encoder.config.hidden_size, d_model), # mean and max are combined hence doubled.
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.1)
        )
        
        # Partial fine-tuning (last 4 layers)
        for param in self.text_encoder.parameters():
            param.requires_grad = False  # Freeze all
        for layer in self.text_encoder.encoder.layer[-4:]:
            for param in layer.parameters():
                param.requires_grad = True  # Unfreeze last 4 [Emperical studies suggest 2-4 for optimal Transfer Learning tasks.]
                
        # Always keep LayerNorms trainable [Suggestions]
        for name, param in self.text_encoder.named_parameters():
            if "LayerNorm" in name:
                param.requires_grad = True

    def forward(self, prompts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            prompts, 
            padding=True, 
            max_length=64,
            truncation=True,
            return_tensors="pt"
        ).to(self.text_encoder.device)

        outputs = self.text_encoder(**inputs)
        
        # Better pooling: Mean + Max pooling
        hidden_states = outputs.last_hidden_state
        mean_pool = hidden_states.mean(dim=1)
        max_pool = hidden_states.max(dim=1).values
        combined = torch.cat([mean_pool, max_pool], dim=1)
        
        return self.pooler(combined)
    
    
class TokenEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        vocab_size: int,
        # num_styles: int = None,  #speaker profile
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim_model = dim_model

        self.dropout = torch.nn.Dropout(p=dropout)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.dim_model)

        #speaker profile
        # self.style_embeddings = (
        #     nn.Embedding(num_styles, self.dim_model) if num_styles else None
        # )


    @property
    def weight(self) -> torch.Tensor:
        return self.word_embeddings.weight

    def embedding(self, index: int) -> torch.Tensor:
        return self.word_embeddings.weight[index : index + 1]

    def forward(self, x: torch.Tensor, style_id: torch.Tensor = None):
        """
        Args:
          x: Input tensor for text tokens 
          style_id: Tensor of style IDs for conditioning.
        """
        X = self.word_embeddings(x)

        # # Add style embedding if provided
        # if self.style_embeddings and style_id is not None:
        #     style_emb = self.style_embeddings(style_id)  # Shape: [Batch, dim_model]
        #     X = X + style_emb.unsqueeze(1)  # Broadcast style_emb across all time steps
        
        X = self.dropout(X)

        return X
    






class SinePositionalEmbedding(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dropout: float = 0.0,
        scale: bool = False,
        alpha: bool = False,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.x_scale = math.sqrt(dim_model) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.dropout = torch.nn.Dropout(p=dropout)

        self.reverse = False
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, 4000))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.dim_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(
                0, x.size(1), dtype=torch.float32
            ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.dim_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.extend_pe(x)
        output = x.unsqueeze(-1) if x.ndim == 2 else x
        output = output * self.x_scale + self.alpha * self.pe[:, : x.size(1)]
        return self.dropout(output)