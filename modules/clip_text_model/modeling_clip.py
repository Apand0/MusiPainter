# @title modules/clip_text_model/modeling_clip.py
"""
modeling_clip.py — Modified CLIP text model for Musipainter audio injection.

Based on the HuggingFace Transformers CLIP implementation (Apache 2.0).
Key modifications vs the original:
  - CLIPTextEmbeddings.forward() accepts an audio_e tensor and injects it at
    the <*> placeholder position via a differentiable linear combination
    (preserves gradient flow to FGAEmbedder).
  - CLIPTextModel.forward() has audio_e as its first positional argument;
    an integer-dtype guard makes it backward-compatible with the Diffusers
    StableDiffusionPipeline which passes input_ids positionally.
  - EOS pooling uses a saved copy of input_ids where the placeholder ID has
    been lowered to 5, so argmax() reliably finds the EOS token.
"""

# coding=utf-8
# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "openai/clip-vit-base-patch32"

CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai/clip-vit-base-patch32",
    # See all CLIP models at https://huggingface.co/models?filter=clip
]


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """Expand attention_mask from [bsz, seq_len] to [bsz, 1, tgt_seq_len, src_seq_len]."""
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


@dataclass
class CLIPTextModelOutput(ModelOutput):
    """
    Output type for the CLIP text model, including an optional pooled representation.

    Args:
        text_embeds: projection-layer output (only when model is initialised with_projection=True).
        last_hidden_state: sequence of hidden states from the final encoder layer.
        hidden_states: hidden states from every layer (optional).
        attentions: attention weights from every layer (optional).
    """
    text_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class CLIPTextEmbeddings(nn.Module):
    """
    CLIP token + position embeddings, extended to inject an audio embedding at
    the <*> placeholder position during training.

    The placeholder token ID is set explicitly via MusicTokenWrapper.set_placeholder_token_id()
    after tokenizer.add_tokens("<*>"). A fallback to input_ids.max() is kept for
    backward-compatibility with older checkpoints but is unreliable and should not
    be relied on for new training runs.
    """

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding    = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids is contiguous in memory and exported when serialised.
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )

        # Set by MusicTokenWrapper.set_placeholder_token_id() before any forward pass.
        self.placeholder_token_id: Optional[int] = None

    def forward(
        self,
        audio_e=None,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            # Resolve which token ID marks the <*> placeholder.
            if self.placeholder_token_id is not None:
                placeholder_id = self.placeholder_token_id
            else:
                # Legacy fallback: unreliable when other tokens have higher IDs.
                placeholder_id = input_ids.max().item()

            # Always work on a clone: input_ids is reused by CLIPTextTransformer
            # for EOS detection via argmax() and must not be modified in-place.
            input_ids_safe = input_ids.clone()
            indices = torch.where(input_ids_safe == placeholder_id)

            if audio_e is not None:
                # --- Training mode: inject audio embedding at placeholder position ---
                #
                # Replace the placeholder with token_id=5 (a safe, common vocab token)
                # before the embedding lookup, then blend the audio vector back in via
                # a differentiable mask. This is required because an in-place assignment
                # `inputs_embeds[indices] = audio_e` would break the autograd graph:
                # base_embeds has requires_grad=False (frozen embedding table), so
                # autograd would not create an edge to audio_e.
                #
                # The linear combination keeps the gradient path intact:
                #   inputs_embeds = base * (1 - mask) + audio_expanded * mask
                # where mask=1 only at the placeholder position.
                # Forward value: identical to direct injection (mask is binary 0/1).
                # Backward: loss → encoder_hidden_states → inputs_embeds →
                #           audio_expanded → audio_e → FGAEmbedder. Full chain.
                input_ids_for_embed = input_ids_safe.masked_fill(
                    input_ids_safe == placeholder_id, 5
                )
                base_embeds = self.token_embedding(input_ids_for_embed)
                # [B, seq_len, D], requires_grad=False (frozen)

                audio_e = audio_e.to(dtype=base_embeds.dtype)
                audio_e_squeezed = audio_e.squeeze(1) if (
                    audio_e.dim() == 3 and audio_e.shape[1] == 1
                ) else audio_e
                # [B, D], requires_grad=True

                placeholder_mask = (
                    (input_ids_safe == placeholder_id)
                    .to(dtype=base_embeds.dtype)
                    .unsqueeze(-1)
                )  # [B, seq_len, 1] — 1.0 at placeholder, 0.0 elsewhere

                audio_expanded = audio_e_squeezed.unsqueeze(1).expand(
                    -1, base_embeds.size(1), -1
                )
                # [B, seq_len, D] — broadcast; only selected at placeholder position by mask

                inputs_embeds = (
                    base_embeds * (1.0 - placeholder_mask)
                    + audio_expanded * placeholder_mask
                )
                # [B, seq_len, D], requires_grad=True via audio_expanded

            else:
                # --- Inference mode ---
                # The embedding table has already been resized in the test script
                # to contain the audio embedding at the placeholder position.
                inputs_embeds = self.token_embedding(input_ids_safe)

            # Save a version of input_ids where the placeholder ID has been lowered to 5.
            # CLIPTextTransformer uses it for EOS pooling (argmax), so the placeholder
            # must not be the maximum ID in the sequence. Stored as a temporary attribute;
            # thread-safe because each forward is synchronous and DDP uses separate replicas.
            input_ids_safe[input_ids_safe == placeholder_id] = 5
            self._last_input_ids_for_pooling = input_ids_safe

        position_embeddings = self.position_embedding(position_ids)
        return inputs_embeds + position_embeddings


class CLIPAttention(nn.Module):
    """Multi-headed self-attention from 'Attention Is All You Need'."""

    def __init__(self, config):
        super().__init__()
        self.config    = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim  = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim={self.embed_dim}, "
                f"num_heads={self.num_heads})."
            )
        self.scale   = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj   = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj   = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj   = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scale
        key_states   = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape   = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states   = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len      = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, "
                f"but is {attn_weights.size()}"
            )

        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, "
                    f"but is {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, "
                    f"but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # Reshape twice to keep attn_weights in the autograd graph.
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights          = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs  = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, "
                f"but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class CLIPMLP(nn.Module):
    """Two-layer feed-forward network used inside each CLIP encoder layer."""

    def __init__(self, config):
        super().__init__()
        self.config        = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1           = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2           = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    """Single transformer block: LayerNorm → Self-Attention → residual → LayerNorm → MLP → residual."""

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.embed_dim   = config.hidden_size
        self.self_attn   = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp         = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states:          [batch, seq_len, embed_dim]
            attention_mask:         [batch, 1, tgt_len, src_len] — padding mask
            causal_attention_mask:  [batch, 1, tgt_len, src_len] — causal mask
            output_attentions:      whether to return attention weights
        """
        residual     = hidden_states
        hidden_states = hidden_states.to(self.layer_norm1.weight.dtype)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states  # first residual connection

        residual      = hidden_states
        hidden_states = hidden_states.to(self.layer_norm2.weight.dtype)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states  # second residual connection

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class CLIPPreTrainedModel(PreTrainedModel):
    """Abstract base for CLIP models: weight initialisation and pretrained-model loading."""

    config_class             = CLIPConfig
    base_model_prefix        = "clip"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        factor = self.config.initializer_factor
        if isinstance(module, CLIPTextEmbeddings):
            module.token_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.position_embedding.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, CLIPAttention):
            in_proj_std  = (module.embed_dim ** -0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim ** -0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, CLIPMLP):
            in_proj_std = (
                (module.config.hidden_size ** -0.5)
                * ((2 * module.config.num_hidden_layers) ** -0.5)
                * factor
            )
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, CLIPEncoder):
            module.gradient_checkpointing = value


CLIP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`CLIPConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class CLIPEncoder(nn.Module):
    """
    Transformer encoder: a stack of config.num_hidden_layers CLIPEncoderLayer blocks.

    Args:
        config: CLIPConfig
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict          = return_dict          if return_dict          is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions    else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states, attention_mask, causal_attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states, attention_mask, causal_attention_mask,
                    output_attentions=output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class CLIPTextTransformer(nn.Module):
    """Full CLIP text transformer: embeddings → encoder stack → layer norm → EOS pooling."""

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config      = config
        embed_dim        = config.hidden_size
        self.embeddings  = CLIPTextEmbeddings(config)
        self.encoder     = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
    def forward(
        self,
        audio_e=None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""Returns:"""
        output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict          = return_dict          if return_dict          is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids   = input_ids.view(-1, input_shape[-1])

        # Run the modified embedding layer, which injects audio_e at the placeholder position.
        hidden_states = self.embeddings(audio_e, input_ids=input_ids, position_ids=position_ids)

        bsz, seq_len = input_shape
        # CLIP uses a causal mask (each token attends only to past tokens).
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(
            bsz, seq_len, hidden_states.dtype
        ).to(hidden_states.device)

        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = last_hidden_state.to(self.final_layer_norm.weight.dtype)
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # EOS pooling: take the hidden state at the position of the EOS token.
        # Use the saved input_ids copy (placeholder lowered to 5) so that argmax()
        # does not confuse the placeholder with the EOS token.
        _ids_for_pool  = getattr(self.embeddings, '_last_input_ids_for_pooling', None)
        ids_for_argmax = _ids_for_pool if _ids_for_pool is not None else input_ids
        pooled_output  = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            ids_for_argmax.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # Upper-triangular additive mask filled with -inf; lower triangle is 0.
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.finfo(dtype).min)
        mask.triu_(1)           # zero out the lower diagonal
        mask = mask.unsqueeze(1)
        return mask


@add_start_docstrings(
    """The text model from CLIP without any head or projection on top.""",
    CLIP_START_DOCSTRING,
)
class CLIPTextModel(CLIPPreTrainedModel):
    """
    Top-level CLIP text model used by Musipainter.

    Extends the standard HuggingFace CLIPTextModel by adding audio_e as the
    first positional argument so the training loop can pass audio embeddings
    without keyword-argument overhead. An integer-dtype guard on audio_e ensures
    backward-compatibility with the Diffusers StableDiffusionPipeline, which
    calls text_encoder(input_ids, attention_mask=...) positionally.
    """

    config_class       = CLIPTextConfig
    _no_split_modules  = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = CLIPTextTransformer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
    def forward(
        self,
        audio_e=None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPTextModel

        >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""

        # Compatibility guard: Diffusers StableDiffusionPipeline calls
        # text_encoder(input_ids, attention_mask=...) positionally.
        # When audio_e arrives as an integer tensor it is actually input_ids.
        if audio_e is not None and audio_e.dtype in (torch.int32, torch.int64, torch.long, torch.int):
            input_ids = audio_e
            audio_e   = None

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
            audio_e,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
