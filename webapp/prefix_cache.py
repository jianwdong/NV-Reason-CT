"""Reuse the encoded scan across questions about the same CT.

Every NV-Reason-CT prompt starts with the same ~13,830 tokens: the opening of the first user
turn and the 13,824 image tokens of the CT. Encoding them is most of the wait before the first
answer token (about 30 s on an M2 Max). `PrefixCache` encodes that prefix once per scan and
region, keeps the model cache it produces, and continues every later prompt from a copy of it,
so only the text after the image (the question and any earlier turns) is encoded again. The
prompt itself is unchanged; only the computation is split at the end of the image.

Qwen3.5 mixes full-attention layers, whose key/value cache simply grows, with Gated DeltaNet
linear-attention layers, which carry a convolution state and a recurrent state. Transformers
5.6.2 continues those states only one token at a time: a multi-token chunk on top of an
existing cache is processed as if the layer had seen nothing before. `enable_chunked_continuation`
patches the linear-attention layers so such a chunk starts from the cached states, using the
same torch kernels the library already uses (both accept an initial state).
"""

from __future__ import annotations

import copy
import threading
import types
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5 as q35


def enable_chunked_continuation(model):
    """Let Gated DeltaNet layers continue from cached state on multi-token input."""
    patched = 0
    for module in model.modules():
        if isinstance(module, q35.Qwen3_5GatedDeltaNet) and not hasattr(module, "_uncached_forward"):
            module._uncached_forward = module.forward
            module.forward = types.MethodType(_gated_delta_forward, module)
            patched += 1
    if not patched:
        raise RuntimeError("No Qwen3.5 Gated DeltaNet layers found; cannot reuse the prompt cache.")
    return patched


def _gated_delta_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    seq_len = hidden_states.shape[1]
    if cache_params is None or seq_len == 1 or not cache_params.has_previous_state(self.layer_idx):
        return self._uncached_forward(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)

    # Same computation as Qwen3_5GatedDeltaNet.forward, but the convolution and the delta rule
    # start from this layer's cached states instead of from zeros.
    hidden_states = q35.apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size = hidden_states.shape[0]
    layer = cache_params.layers[self.layer_idx]

    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    # Causal convolution over [last cached inputs | new chunk]; updates layer.conv_states in place.
    mixed_qkv = q35.torch_causal_conv1d_update(
        mixed_qkv, layer.conv_states, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation
    ).transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta,
        initial_state=layer.recurrent_states,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

    core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    return self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))


@dataclass
class _Entry:
    prefix_ids: torch.Tensor  # (1, n) token ids the cache covers
    cache: DynamicCache
    rope_deltas: torch.Tensor


class PrefixCache:
    """Encoded image prefixes for the most recent scans and regions (about 0.5 GB each)."""

    def __init__(self, model, vision_end_id, max_entries=2):
        enable_chunked_continuation(model)
        self.model = model
        self.vision_end_id = vision_end_id
        self.max_entries = max_entries
        self.entries: OrderedDict[tuple, _Entry] = OrderedDict()
        self.lock = threading.Lock()

    def prefix_length(self, input_ids):
        hits = (input_ids[0] == self.vision_end_id).nonzero()
        if len(hits) == 0:
            raise ValueError("Prompt has no image; nothing to reuse.")
        return int(hits[0]) + 1

    def has(self, key, input_ids):
        with self.lock:
            entry = self.entries.get(key)
        n = self.prefix_length(input_ids)
        return entry is not None and torch.equal(entry.prefix_ids.cpu(), input_ids[:, :n].cpu())

    def drop(self, predicate):
        with self.lock:
            for key in [k for k in self.entries if predicate(k)]:
                del self.entries[key]

    def prepare(self, key, inputs, device):
        """Return generate() kwargs that continue from the cached image prefix.

        Encodes and stores the prefix first if this scan and region are not cached yet.
        Returns ``(kwargs, reused)``. Call while holding the GPU lock.
        """
        input_ids = inputs["input_ids"]
        n = self.prefix_length(input_ids)
        reused = self.has(key, input_ids)
        if not reused:
            self._encode_prefix(key, inputs, n, device)
        with self.lock:
            entry = self.entries[key]
            self.entries.move_to_end(key)
        with torch.inference_mode():
            cache = copy.deepcopy(entry.cache)  # generation extends the copy; the entry stays as encoded
            # Text after the image gets 3D-RoPE positions offset by the image's delta.
            self.model.model.rope_deltas = entry.rope_deltas.clone()
        kwargs = {
            "input_ids": input_ids.to(device),
            "attention_mask": torch.ones_like(input_ids).to(device),
            "past_key_values": cache,
        }
        return kwargs, reused

    def _encode_prefix(self, key, inputs, n, device):
        cache = DynamicCache(config=self.model.config)
        input_ids = inputs["input_ids"][:, :n].to(device)
        mm = inputs.get("mm_token_type_ids")
        mm = (mm[:, :n] if mm is not None else (input_ids == self.model.config.image_token_id).int()).to(device)
        with torch.inference_mode():
            # The base model (no LM head): logits for 13,830 positions would need several GB.
            self.model.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                pixel_values=inputs["pixel_values"].to(device),
                image_grid_thw=inputs["image_grid_thw"].to(device),
                mm_token_type_ids=mm,
                past_key_values=cache,
                use_cache=True,
            )
        entry = _Entry(input_ids.cpu(), cache, self.model.model.rope_deltas.clone())
        with self.lock:
            self.entries[key] = entry
            while len(self.entries) > self.max_entries:
                self.entries.popitem(last=False)
