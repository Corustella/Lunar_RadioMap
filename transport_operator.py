"""Finite latent scene interaction and bounded-cost receiver queries.

Direction slots are learned latent states with geometric direction codes, not
ray records. All states can interact; visibility never removes an edge.
"""

import torch
from torch import nn
from torch.nn import functional as F


class _TransportBlock(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.geometry_qk = nn.Linear(11, width * 2, bias=False)
        self.source = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(),
                                 nn.Linear(width * 4, width))
        self.attention_scale = nn.Parameter(torch.full((width,), .1))
        self.ffn_scale = nn.Parameter(torch.full((width,), .1))
        self.source_scale = nn.Parameter(torch.full((width,), .1))

    def forward(self, states, geometry, source):
        states = states + self.source_scale * self.source(source)[:, None]
        q, k, v = self.qkv(self.norm(states)).chunk(3, dim=-1)
        gq, gk = self.geometry_qk(geometry.to(q.dtype)).chunk(2, dim=-1)
        # Geometry enters Q/K without allocating BxNxNxgeometry tensors.
        attended = F.scaled_dot_product_attention((q + gq)[:, None], (k + gk)[:, None],
                                                  v[:, None], dropout_p=0.0)[:, 0]
        states = states + self.attention_scale * self.output(attended)
        return states + self.ffn_scale * self.ffn(self.ffn_norm(states))


class SceneTransport(nn.Module):
    """1024 states, four interactions, 4096 receiver queries with 32 keys each.

Dense latent attention costs O(B*L*N^2*d); a backend may materialize O(B*N^2)
attention. Receiver readout costs O(B*Q*K*d), not O(B*65536*N*d).
"""
    def __init__(self):
        super().__init__()
        self.surface = nn.Conv2d(256, 64, 1)
        self.deep = nn.Conv2d(512, 64, 1)
        self.source = nn.Sequential(nn.Linear(256 + 16 + 9, 64), nn.GELU(), nn.Linear(64, 64))
        self.initial_geometry = nn.Linear(11, 64)
        self.direction_embedding = nn.Parameter(torch.empty(4, 64))
        nn.init.normal_(self.direction_embedding, std=.02)
        self.blocks = nn.ModuleList([_TransportBlock() for _ in range(4)])
        self.global_queries = nn.Parameter(torch.empty(16, 64))
        nn.init.normal_(self.global_queries, std=.02)
        self.global_source = nn.Linear(64, 64)
        self.global_kv = nn.Linear(64, 128)
        self.receiver_query = nn.Conv2d(64 + 9, 64, 1)
        self.receiver_kv = nn.Linear(64, 128)
        self.receiver_output = nn.Linear(64, 64)
        self.receiver_scale = nn.Parameter(torch.full((64,), .1))
        self.global_norm = nn.LayerNorm(64)
        self.state_norm = nn.LayerNorm(64)
        self.register_buffer("directions", torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]), persistent=False)
        # Grid centres follow the encoder's actual stride, including borders.
        row, col = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
        coarse_row, coarse_col = row.float() / 4, col.float() / 4
        r0, c0 = coarse_row.long(), coarse_col.long()
        indices = []
        for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)):
            patch = (r0 + dr).clamp_max(15) * 16 + (c0 + dc).clamp_max(15)
            indices.append(patch.flatten()[:, None] * 4 + torch.arange(4)[None])
        indices = torch.cat(indices, dim=1)
        # Clamped last-strip neighbours may coincide: do not count duplicates.
        unique = torch.ones_like(indices, dtype=torch.bool)
        for i in range(1, indices.shape[1]):
            unique[:, i] = ~(indices[:, i:i+1] == indices[:, :i]).any(1)
        self.register_buffer("receiver_indices", indices, persistent=False)
        self.register_buffer("receiver_unique", unique, persistent=False)

    @staticmethod
    def _source_sample(surface, full, geometry, tx_index):
        batch = surface.shape[0]
        row = (tx_index // 256).float()
        col = (tx_index % 256).float()
        grid = torch.stack((2 * col / 240 - 1, 2 * row / 240 - 1), dim=-1)[:, None, None]
        # Use the stride-16 lattice (0..240), not a resized endpoint grid.
        with torch.autocast(device_type=surface.device.type, enabled=False):
            context = F.grid_sample(surface.float(), grid, mode="bilinear",
                                    padding_mode="border", align_corners=True).flatten(1)
        fine = full.flatten(2).gather(2, tx_index[:, None, None].expand(batch, 16, 1)).squeeze(-1)
        geo = geometry.flatten(2).gather(2, tx_index[:, None, None].expand(batch, 9, 1)).squeeze(-1)
        return torch.cat((context.to(fine.dtype), fine, geo.to(fine.dtype)), dim=1)

    def encode(self, surface16, deep8, full256, geometry, tx_index):
        # AMP otherwise erases the early small transport/query gradients.
        # The convolutional encoder/decoder still use the requested AMP mode.
        with torch.autocast(device_type=surface16.device.type, enabled=False):
            return self._encode(surface16.float(), deep8.float(), full256.float(), geometry, tx_index)

    def _encode(self, surface16, deep8, full256, geometry, tx_index):
        patch = self.surface(surface16) + F.interpolate(self.deep(deep8), size=(16, 16),
                                                        mode="bilinear", align_corners=False)
        source = self.source(self._source_sample(surface16, full256, geometry, tx_index))
        geo = geometry[:, :, ::16, ::16].flatten(2).transpose(1, 2)
        batch = patch.shape[0]
        direction = self.directions[None, None].expand(batch, 256, -1, -1)
        geo = torch.cat((geo[:, :, None].expand(-1, -1, 4, -1), direction), dim=-1).reshape(batch, 1024, 11)
        states = patch.flatten(2).transpose(1, 2)[:, :, None] + self.direction_embedding[None, None]
        states = states.reshape(batch, 1024, 64) + self.initial_geometry(geo.to(patch.dtype))
        for block in self.blocks:
            states = block(states, geo, source)
        return states, source

    def query(self, receiver64, geometry, states, source):
        with torch.autocast(device_type=receiver64.device.type, enabled=False):
            return self._query(receiver64.float(), geometry, states.float(), source.float())

    def _query(self, receiver64, geometry, states, source):
        batch = states.shape[0]
        k, v = self.global_kv(self.state_norm(states)).chunk(2, dim=-1)
        q = self.global_queries[None] + self.global_source(source)[:, None]
        global_states = F.scaled_dot_product_attention(q[:, None].to(k.dtype), k[:, None], v[:, None],
                                                       dropout_p=0.0)[:, 0]
        global_states = self.global_norm(global_states)
        query = self.receiver_query(torch.cat((receiver64, geometry[:, :, ::4, ::4].to(receiver64.dtype)), dim=1))
        query = query.flatten(2).transpose(1, 2)
        allowed = torch.cat((self.receiver_unique, torch.ones(4096, 16, dtype=torch.bool,
                                                             device=states.device)), dim=1)
        # Chunking also keeps the CUDA SDPA batch dimension below launch limits
        # at batch16 (flattening all4096 queries would create 65536 batches).
        chunks = []
        for start in range(0, 4096, 512):
            local = states[:, self.receiver_indices[start:start+512]]
            context = torch.cat((local, global_states[:, None].expand(-1, 512, -1, -1)), dim=2)
            k, v = self.receiver_kv(context).chunk(2, dim=-1)
            q = query[:, start:start+512].reshape(batch * 512, 1, 1, 64)
            k, v = (t.reshape(batch * 512, 1, 32, 64) for t in (k, v))
            mask = allowed[start:start+512][None].expand(batch, -1, -1).reshape(batch * 512, 1, 1, 32)
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
            chunks.append(attended.reshape(batch, 512, 64))
        context = self.receiver_output(torch.cat(chunks, dim=1))
        update = context * self.receiver_scale
        # A direct feature path to the readout prevents remote context from
        # receiving gradients only through several small decoder convolutions.
        context = context.transpose(1, 2).reshape(batch, 64, 64, 64)
        return receiver64 + update.transpose(1, 2).reshape(batch, 64, 64, 64), context
