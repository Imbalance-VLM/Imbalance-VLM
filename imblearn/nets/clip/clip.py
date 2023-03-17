from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from typing import Any, Optional, Tuple, Union

from transformers import CLIPPreTrainedModel, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionTransformer
from timm.models.vision_transformer import Block

class CLIPVisionModelWithProjection(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    def __init__(self, config: CLIPVisionConfig, num_classes, decoder_num_heads, mlp_ratio, decoder_depth):
        super().__init__(config)
        # print(config)

        self.vision_model = CLIPVisionTransformer(config)

        prev_dim = config.hidden_size

        if decoder_depth > 0:
            self.decoder_blocks = nn.Sequential(*[Block(prev_dim, decoder_num_heads, mlp_ratio, qkv_bias=True) for i in range(decoder_depth)])
        else:
            self.decoder_blocks = None
        
        self.classifier = nn.Linear(prev_dim, num_classes)

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        x: Optional[torch.FloatTensor] = None,
        only_fc: Optional[bool] = False, 
        only_feat: Optional[bool] = False,
    ):

        if only_fc:
            return self.classifier(x)

        vision_outputs = self.vision_model(pixel_values=x, output_hidden_states=False).last_hidden_state

        if self.decoder_blocks is not None:
            vision_outputs = self.decoder_blocks(vision_outputs)
        
        pooled_output = vision_outputs[:, 0, :]
        
        if only_feat:
            return pooled_output

        image_embeds = self.classifier(pooled_output)

        return {
            'logits': image_embeds,
            'feat': pooled_output,
        }


def openai_clip_vit_large_patch14(**kwargs):
    """ openai/clip-vit-large-patch14
    """
    print(kwargs)
    model = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14', num_classes=kwargs['num_classes'], decoder_num_heads=kwargs['decoder_num_heads'], mlp_ratio=kwargs['decoder_mlp_ratio'], decoder_depth=kwargs['decoder_depth'])
    return model

def openai_clip_vit_large_patch14_336(**kwargs):
    """ openai/clip-vit-large-patch14-336
    """
    model = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14-336', num_classes=kwargs['num_classes'])
    return model


def openai_clip_vit_base_patch16(**kwargs):
    """ openai/clip-vit-base-patch16
    """
    model = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-base-patch16', num_classes=kwargs['num_classes'])
    return model

def openai_clip_vit_base_patch32(**kwargs):
    """ openai/clip-vit-base-patch32
    """
    model = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch32', num_classes=kwargs['num_classes'])
    return model
