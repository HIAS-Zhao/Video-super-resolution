class Mlp(nn.Module):
    """Swin-style FFN: 1x1 conv -> GELU -> 1x1 conv"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GuidedMHOffsetPredictor(nn.Module):
    """
    Output: residual offsets in pixel units
    shape: (B, N, heads, K, H, W, 2)
    """

    def __init__(self, in_channels, hidden=128, heads=8, K=9, max_residue_magnitude=10.0):
        super().__init__()
        self.heads = heads
        self.K = K
        self.max_res = float(max_residue_magnitude)

        # Use 4C inputs: ref, supp, diff, absdiff (strongly recommended)
        in_ch = in_channels * 4

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, heads * K * 2, 1)  # K (dx, dy) pairs for each head
        )
        self.init_offset()

    def init_offset(self):
        # Initialize residuals to 0 for more stable training (same idea as GuidedDeformAttnPack)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, ref_feat, supp_feats):
        """
        ref_feat:  (B, C, H, W)
        supp_feats:(B, N, C, H, W)  coarse-aligned support features
        """
        B, C, H, W = ref_feat.shape
        N = supp_feats.shape[1]

        ref = ref_feat.unsqueeze(1).expand(B, N, C, H, W).reshape(B * N, C, H, W)
        supp = supp_feats.reshape(B * N, C, H, W)

        diff = ref - supp
        x = torch.cat([ref, supp, diff, diff.abs()], dim=1)  # (B*N, 4C, H, W)

        raw = self.net(x)  # (B*N, heads*K*2, H, W)
        raw = raw.view(B, N, self.heads, self.K, 2, H, W).permute(0, 1, 2, 3, 5, 6, 4).contiguous()
        # -> (B, N, heads, K, H, W, 2)

        # Residual offsets in pixel units, clamped to [-max_res, max_res]
        res_pix = self.max_res * torch.tanh(raw / self.max_res)
        return res_pix


class SpatialTemporalTransformer(nn.Module):
    def __init__(
            self,
            feat_channels,
            encoding_dim=64,
            num_heads=8,
            mlp_ratio=2,
            load_path=None,
            add_ref_residual: bool = True,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.feat_channels = feat_channels
        self.encoding_dim = encoding_dim
        self.num_heads = num_heads
        self.K = 9
        self.head_dim = encoding_dim // num_heads

        self.add_ref_residual = bool(add_ref_residual)

        self.norm1 = nn.LayerNorm(feat_channels)
        self.norm2 = nn.LayerNorm(feat_channels)

        self.mlp = Mlp(feat_channels, feat_channels * mlp_ratio, feat_channels, drop=dropout)

        # Each head predicts a scalar relative-position bias, which pairs well with per-head offsets
        self.rel_mlp = Mlp(3, 32 * mlp_ratio, 1, drop=dropout)

        self.q_proj = nn.Linear(feat_channels, encoding_dim, bias=True)
        self.kv_proj = nn.Linear(feat_channels, encoding_dim * 2, bias=True)
        self.proj = nn.Linear(encoding_dim, feat_channels)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Guided multi-head multi-hypothesis offsets, max_residue_magnitude=10
        self.offset_predictor = GuidedMHOffsetPredictor(
            in_channels=feat_channels, hidden=64, heads=num_heads, K=self.K, max_residue_magnitude=10.0
        )

    def pix2norm(self, res_pix, H, W):
        # res_pix: (..., 2) in pixel units
        scale_x = 2.0 / (W - 1) if W > 1 else 0.0
        scale_y = 2.0 / (H - 1) if H > 1 else 0.0
        out = res_pix.clone()
        out[..., 0] = out[..., 0] * scale_x
        out[..., 1] = out[..., 1] * scale_y
        return out

    def _coerce_flow_to_bnhw2(self, supp_flows, B: int, N: int, H: int, W: int, device, dtype):
        """Normalize different flow container formats to (B, N, H, W, 2) in pixel units."""
        if supp_flows is None:
            return None

        if isinstance(supp_flows, torch.Tensor):
            flow = supp_flows
            if flow.dim() != 5:
                raise ValueError(f'supp_flows tensor must be 5D, got shape={tuple(flow.shape)}')
            if flow.shape[0] != B or flow.shape[1] != N:
                raise ValueError(f'supp_flows tensor must have shape (B,N,*,*,*), got {tuple(flow.shape)}')
            if flow.shape[-1] == 2:
                # (B,N,H,W,2)
                flow_bnhw2 = flow
            elif flow.shape[2] == 2:
                # (B,N,2,H,W) -> (B,N,H,W,2)
                flow_bnhw2 = flow.permute(0, 1, 3, 4, 2).contiguous()
            else:
                raise ValueError(
                    f'supp_flows tensor must be (B,N,2,H,W) or (B,N,H,W,2), got {tuple(flow.shape)}'
                )
        else:
            # Sequence[Tensor] of length N, each element is (B,2,H,W) or (B,H,W,2)
            if len(supp_flows) != N:
                raise ValueError(f'supp_flows length ({len(supp_flows)}) must match supp_feats length ({N}).')
            flows_list = []
            for i, f in enumerate(supp_flows):
                if not isinstance(f, torch.Tensor):
                    raise TypeError(f'supp_flows[{i}] must be a Tensor, got {type(f)}')
                if f.dim() != 4:
                    raise ValueError(f'supp_flows[{i}] must be 4D, got shape={tuple(f.shape)}')
                if f.shape[0] != B:
                    raise ValueError(f'supp_flows[{i}] batch mismatch: expected B={B}, got {f.shape[0]}')
                if f.shape[-1] == 2:
                    flow_i = f
                elif f.shape[1] == 2:
                    flow_i = f.permute(0, 2, 3, 1).contiguous()
                else:
                    raise ValueError(
                        f'supp_flows[{i}] must be (B,2,H,W) or (B,H,W,2), got {tuple(f.shape)}'
                    )
                flows_list.append(flow_i)
            flow_bnhw2 = torch.stack(flows_list, dim=1)

        if flow_bnhw2.shape[2] != H or flow_bnhw2.shape[3] != W:
            raise ValueError(
                f'supp_flows spatial size must match ref_feat: expected (H,W)=({H},{W}), '
                f'got ({flow_bnhw2.shape[2]},{flow_bnhw2.shape[3]})'
            )
        return flow_bnhw2.to(device=device, dtype=dtype)

    def _make_base_grid(self, H, W, device, dtype):
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        base = torch.stack((xx, yy), dim=-1)  # (H,W,2) in (x,y) order
        return base.view(1, 1, 1, 1, H, W, 2)  # (1,1,1,1,H,W,2)

    def deformable_attention(
            self,
            ref_feat,
            supp_feats,
            supp_flows: Optional[Union[Sequence[torch.Tensor], torch.Tensor]] = None,
            supp_feats_for_offset: Optional[Sequence[torch.Tensor]] = None,
            supp_dts: Optional[Union[Sequence[int], torch.Tensor]] = None,
    ):
        B, C, H, W = ref_feat.shape
        N = len(supp_feats)

        if N == 0:
            return ref_feat.permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)

        all_feats = [ref_feat] + supp_feats
        all_fused = torch.stack(all_feats, dim=1).permute(0, 1, 3, 4, 2).contiguous()  # (B,N+1,H,W,C)
        norm_feats = self.norm1(all_fused)

        # Q: (B, heads, H, W, dh)
        Q = self.q_proj(norm_feats[:, 0])  # (B,H,W,enc)
        Q = Q.view(B, H, W, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4).contiguous()

        # K,V maps: (B, N, enc, H, W)
        kv = self.kv_proj(norm_feats[:, 1:])  # (B,N,H,W,2enc)
        kv = kv.view(B, N, H, W, 2, self.num_heads, self.head_dim).permute(0, 1, 4, 5, 6, 2, 3).contiguous()
        # -> (B,N,2,heads,dh,H,W)
        K = kv[:, :, 0]  # (B,N,heads,dh,H,W)
        V = kv[:, :, 1]  # (B,N,heads,dh,H,W)

        # --- Offsets in pixel units: residual + optional coarse flow ---
        # Add them in pixel space first, then normalize once.
        flow_bnhw2 = self._coerce_flow_to_bnhw2(supp_flows, B, N, H, W, ref_feat.device, ref_feat.dtype)
        if supp_feats_for_offset is None:
            supp_feats_for_offset = supp_feats
        if len(supp_feats_for_offset) != N:
            raise ValueError(
                f'supp_feats_for_offset length ({len(supp_feats_for_offset)}) must match supp_feats length ({N}).')

        supp_stack = torch.stack(list(supp_feats_for_offset), dim=1)  # (B,N,C,H,W)
        res_pix = self.offset_predictor(ref_feat, supp_stack)  # (B,N,heads,K,H,W,2)
        if flow_bnhw2 is None:
            total_pix = res_pix
        else:
            flow_pix = flow_bnhw2.unsqueeze(2).unsqueeze(3)  # (B,N,1,1,H,W,2)
            total_pix = res_pix + flow_pix.expand(B, N, self.num_heads, self.K, H, W, 2)
        total_norm = self.pix2norm(total_pix, H, W)  # (B,N,heads,K,H,W,2)

        base_grid = self._make_base_grid(H, W, ref_feat.device, ref_feat.dtype).expand(B, N, self.num_heads, self.K, H,
                                                                                       W, 2)
        grid = base_grid + total_norm  # (B,N,heads,K,H,W,2)
        grid = grid.view(B * N * self.num_heads * self.K, H, W, 2)

        # --- Relative bias: (dx, dy, t) -> per-head scalar bias ---
        if supp_dts is None:
            # Backward-compatible path: only indicates ordering
            t = torch.arange(1, N + 1, device=ref_feat.device, dtype=ref_feat.dtype) / (N + 1e-6)
        else:
            if isinstance(supp_dts, torch.Tensor):
                dt = supp_dts.to(device=ref_feat.device)
            else:
                dt = torch.tensor(list(supp_dts), device=ref_feat.device)
            dt = dt.to(dtype=ref_feat.dtype).view(-1)
            if dt.numel() != N:
                raise ValueError(f'supp_dts length ({dt.numel()}) must match supp_feats length ({N}).')
            denom = dt.abs().max().clamp(min=1.0)
            t = dt / (denom + 1e-6)  # [-1, 1]

        t = t.view(1, N, 1, 1, 1, 1, 1).expand(B, N, self.num_heads, self.K, H, W, 1)
        rel = torch.cat([total_norm, t], dim=-1)  # (B,N,heads,K,H,W,3)

        # (B*heads*H*W, N*K, 3) -> bias (B,heads,H,W,N*K)
        rel = rel.permute(0, 2, 4, 5, 1, 3, 6).contiguous().view(B * self.num_heads * H * W, N * self.K, 3)
        rel_bias = self.rel_mlp(rel).squeeze(-1)  # (B*heads*H*W, N*K)
        rel_bias = rel_bias.view(B, self.num_heads, H, W, N * self.K)

        # --- Prepare maps for grid_sample ---
        # K,V: (B,N,heads,dh,H,W) -> (B*N*heads*K, dh, H, W)
        k_maps = K.permute(0, 1, 2, 3, 4, 5).contiguous()  # (B,N,heads,dh,H,W)
        v_maps = V.permute(0, 1, 2, 3, 4, 5).contiguous()

        k_maps = k_maps.view(B * N * self.num_heads, self.head_dim, H, W)
        v_maps = v_maps.view(B * N * self.num_heads, self.head_dim, H, W)

        # Expand the K dimension
        k_maps = k_maps.unsqueeze(1).expand(-1, self.K, -1, -1, -1).contiguous().view(B * N * self.num_heads * self.K,
                                                                                      self.head_dim, H, W)
        v_maps = v_maps.unsqueeze(1).expand(-1, self.K, -1, -1, -1).contiguous().view(B * N * self.num_heads * self.K,
                                                                                      self.head_dim, H, W)

        sampled_k = F.grid_sample(k_maps, grid, mode='bilinear', padding_mode='border', align_corners=True)
        sampled_v = F.grid_sample(v_maps, grid, mode='bilinear', padding_mode='border', align_corners=True)

        # Reshape to (B,heads,H,W,N*K,dh)
        sampled_k = sampled_k.view(B, N, self.num_heads, self.K, self.head_dim, H, W).permute(0, 2, 5, 6, 1, 3,
                                                                                              4).contiguous()
        sampled_v = sampled_v.view(B, N, self.num_heads, self.K, self.head_dim, H, W).permute(0, 2, 5, 6, 1, 3,
                                                                                              4).contiguous()
        sampled_k = sampled_k.view(B, self.num_heads, H, W, N * self.K, self.head_dim)
        sampled_v = sampled_v.view(B, self.num_heads, H, W, N * self.K, self.head_dim)

        # Attention
        attn_score = torch.matmul(Q.unsqueeze(-2), sampled_k.transpose(-1, -2)) / math.sqrt(
            self.head_dim)  # (B,heads,H,W,1,NK)
        attn_score = attn_score + rel_bias.unsqueeze(-2)  # (B,heads,H,W,1,NK)
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
        v_weighted = torch.matmul(attn, sampled_v).squeeze(-2)  # (B,heads,H,W,dh)

        fused = v_weighted.permute(0, 2, 3, 1, 4).contiguous().view(B, H, W, self.num_heads * self.head_dim)
        out = self.proj_drop(self.proj(fused))
        if self.add_ref_residual:
            out = out + all_fused[:, 0]  # Add the residual back to the Q stream (current frame)
        return out  # (B,H,W,C)

    def forward(
            self,
            ref_feat,
            supp_feats,
            supp_flows: Optional[Union[Sequence[torch.Tensor], torch.Tensor]] = None,
            supp_feats_for_offset: Optional[Sequence[torch.Tensor]] = None,
            supp_dts: Optional[Union[Sequence[int], torch.Tensor]] = None,
    ):
        x = self.deformable_attention(
            ref_feat,
            supp_feats,
            supp_flows=supp_flows,
            supp_feats_for_offset=supp_feats_for_offset,
            supp_dts=supp_dts,
        )  # (B,H,W,C)
        x = x + self.mlp(self.norm2(x))
        return x.permute(0, 3, 1, 2).contiguous()