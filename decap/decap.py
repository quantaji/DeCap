from typing import Tuple

import torch
import torch.nn.functional as f
from clip.simple_tokenizer import SimpleTokenizer
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel


class MLP(nn.Module):

    def __init__(self, sizes: Tuple[int, ...], bias: bool = True, act: nn.Module = nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.torch.Tensor) -> torch.torch.Tensor:
        return self.model(x)


class DeCap(nn.Module):

    def __init__(self, prefix_size: int = 512):
        super(DeCap, self).__init__()

        # decoder: 4 layers transformer with 4 attention heads, the decoder is not pretrained
        config = GPT2Config(
            n_layer=4,
            n_head=4,
            activation_function="gelu_new",
            transformers_version="4.30.2",
            task_specific_params={"text-generation": {
                "do_sample": True,
                "max_length": 50
            }},
        )
        self.decoder = GPT2LMHeadModel(config)
        self.tokenizer = SimpleTokenizer()

        self.embedding_size = self.decoder.transformer.wte.weight.shape[1]
        self.clip_project = MLP((prefix_size, self.embedding_size))

    def forward(self, clip_features: torch.Tensor, tokens: torch.Tensor):

        embedding_text = self.decoder.transformer.wte(tokens)
        embedding_clip = self.clip_project(clip_features).reshape(-1, 1, self.embedding_size)
        embedding_cat = torch.cat([embedding_clip, embedding_text], dim=1)

        return self.decoder(inputs_embeds=embedding_cat)

    def decode(self, clip_features: torch.Tensor, entry_length: int = 30):

        embedding_cat = self.clip_project(clip_features).reshape(1, 1, -1)

        tokens = None
        for i in range(entry_length):

            next_token = self.decoder(inputs_embeds=embedding_cat).logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
            next_token_embed = self.decoder.transformer.wte(next_token)

            tokens = torch.cat((tokens, next_token), dim=1) if tokens is not None else next_token

            if next_token.item() == 49407:
                break

            embedding_cat = torch.cat((embedding_cat, next_token_embed), dim=1)

        try:
            output_list = list(tokens.squeeze().cpu().numpy())
            return self.tokenizer.decode(output_list)
        except:
            return 'None'
