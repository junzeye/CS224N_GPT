import torch

from einops import rearrange
from torch import nn
from torch.nn import functional as F
import math

class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads # h, number of attention heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # d_k, attention head hidden size
    self.all_head_size = self.num_attention_heads * self.attention_head_size # d, all head size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size) # d -> d_k * h
    self.key = nn.Linear(config.hidden_size, self.all_head_size) # d -> d_k * h
    self.value = nn.Linear(config.hidden_size, self.all_head_size) # d -> d_k * h

    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d') 
    # b: batch size, h: number of attention heads, t: sequence length, d: attention head hidden size
    return proj

  def attention(self, key, query, value, attention_mask):
    # key: [b, h, tk, d_k], query: [b, h, tq, d_k], value: [b, h, tk, d_k] - in actuality, tk = tq = t.
    # tv is omitted b/c (k,v) always appear in pairs. here for the generality
    # attention_mask: [b, 1, 1, t] - it's really the truncation matrix that's being referred to
    b, h, t, d = key.shape
    raw_attn = (query @ key.transpose(-2, -1)) * (
      1.0 / math.sqrt(self.attention_head_size))  # [b, h, tq, tk]

    causal_mask = torch.tril(torch.ones(t, t, device=key.device))  # [t, t]
    causal_mask = causal_mask.view(1, 1, t, t)  # [1, 1, tq, tk]
    causal_mask = causal_mask.bool() # convert to bool
    attention_mask = ~(attention_mask.bool()) # true for non-padding tokens
    mask = causal_mask & attention_mask  # [b, 1, tq, tk]

    raw_attn = raw_attn.masked_fill(~mask, float('-inf'))  
    attn = self.dropout(F.softmax(raw_attn, dim=-1))
    o = attn @ value
    o = o.transpose(1, 2).contiguous().view(b, t, h * d)
    return o 
    # NOTE: we apply the output projection & dropout in gpt2_layer.py using `self.attention_dense`
    # - a peculiarity of the assignment design...

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value