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

# Module1
class BiSTA(nn.Module):
    """Bidirectional Spatio-Temporal Aggregator

    Aggregates temporal information from support frames in a bidirectional
    manner (forward/backward relative to reference frame). Features from
    left/right temporal neighborhoods are processed independently through
    a shared spatio-temporal attention module, then concatenated along
    the channel dimension for downstream fusion.

    This design enables symmetric temporal modeling while maintaining
    computational efficiency through parameter sharing.

    Args:
        feat_channels (int): Input feature channels
        encoding_dim (int): Internal encoding dimension for attention
        num_heads (int): Number of attention heads
        mlp_ratio (float): Expansion ratio for MLP block
        load_path (str, optional): Path to pretrained weights
        add_ref_residual (bool): Whether to add reference residual in attention
        dropout (float): Dropout rate for regularization

    Shape:
        - ref_feat: (B, C, H, W)
        - supp_feats: List[(B, C, H, W), ...]
        - Output: (B, 2C, H, W) or (B, 0, H, W) if no support frames
    """

    def __init__(
            self,
            feat_channels: int,
            encoding_dim: int = 64,
            num_heads: int = 8,
            mlp_ratio: float = 2,
            load_path: Optional[str] = None,
            add_ref_residual: bool = True,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.feat_channels = feat_channels
        self.encoding_dim = encoding_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.add_ref_residual = add_ref_residual

        # Shared spatio-temporal aggregation module for both directions
        self.aggregator = SpatialTemporalTransformer(
            feat_channels=feat_channels,
            encoding_dim=encoding_dim,
            num_heads=num_heads // 2,  # Reduce heads for bidirectional sharing
            mlp_ratio=mlp_ratio,
            load_path=load_path,
            add_ref_residual=add_ref_residual,
            dropout=dropout,
        )

    def forward(
            self,
            ref_feat: torch.Tensor,
            supp_feats: Optional[List[torch.Tensor]] = None,
            left_len: Optional[int] = None,
            supp_value_feats: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Bidirectional temporal aggregation with symmetric processing.

        Args:
            ref_feat: Reference frame features (B, C, H, W)
            supp_feats: List of support frame features for aggregation
            left_len: Number of frames to treat as "left" (past) neighborhood.
                     If None, defaults to min(2, len(supp_feats)//2)
            supp_value_feats: Optional separate value features for attention.
                             If None, uses supp_feats directly.

        Returns:
            torch.Tensor: Aggregated features (B, 2C, H, W) for downstream fusion.
                         Returns zero tensor if no valid support frames available.
        """
        # Handle empty support frame list
        if supp_feats is None or len(supp_feats) == 0:
            b, c, h, w = ref_feat.shape
            return ref_feat.new_zeros(b, c * 2, h, w)

        # Determine temporal split point for bidirectional processing
        if left_len is None:
            left_len = min(2, len(supp_feats) // 2)
        left_len = int(max(left_len, 0))

        # Align value features with query features
        if supp_value_feats is None:
            supp_value_feats = supp_feats
        if len(supp_value_feats) != len(supp_feats):
            raise ValueError('supp_value_feats length must match supp_feats length.')

        # Split support frames into bidirectional neighborhoods
        left_feats = supp_feats[:left_len]
        right_feats = supp_feats[left_len:]
        left_value_feats = supp_value_feats[:left_len]
        right_value_feats = supp_value_feats[left_len:]

        # Process each temporal direction independently with shared weights
        x_l = self.aggregator(ref_feat, left_feats, supp_value_feats=left_value_feats) if len(left_feats) > 0 else None
        x_r = self.aggregator(ref_feat, right_feats, supp_value_feats=right_value_feats) if len(
            right_feats) > 0 else None

        # Collect valid directional outputs
        second_supp = []
        if x_l is not None:
            second_supp.append(x_l)
        if x_r is not None:
            second_supp.append(x_r)

        # Handle edge cases: no valid outputs or single direction only
        if len(second_supp) == 0:
            b, c, h, w = ref_feat.shape
            return ref_feat.new_zeros(b, c * 2, h, w)
        if len(second_supp) == 1:
            # Duplicate single-direction output to maintain 2C channel format
            second_supp.append(second_supp[0])

        # Concatenate bidirectional features along channel dimension
        # Output shape: (B, 2C, H, W) ready for fusion with reference features
        return torch.cat(second_supp, dim=1)


# Module2

class OAR(nn.Module):
    def __init__(self, channels, hidden_ratio=0.5, alpha_init=0.05, eps=1e-6):
        super().__init__()
        hidden_channels = max(8, int(channels * hidden_ratio))
        self.eps = eps

        self.scharr_x = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.scharr_y = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.context = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True))

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid())

        self.alpha = nn.Parameter(torch.full((1, channels, 1, 1), alpha_init))
        self._init_kernels(channels)

    def _init_kernels(self, channels):
        scharr_x = torch.tensor(
            [[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], dtype=torch.float32)
        scharr_y = torch.tensor(
            [[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], dtype=torch.float32)
        gaussian = torch.tensor(
            [[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], dtype=torch.float32)
        gaussian = gaussian / gaussian.sum()

        with torch.no_grad():
            self.scharr_x.weight.copy_(scharr_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
            self.scharr_y.weight.copy_(scharr_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
            self.context.weight.copy_(gaussian.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))

        self.scharr_x.weight.requires_grad = False
        self.scharr_y.weight.requires_grad = False

    def forward(self, x):
        edge_x = self.scharr_x(x)
        edge_y = self.scharr_y(x)
        edge = torch.sqrt(edge_x * edge_x + edge_y * edge_y + self.eps)
        context = self.context(x)

        detail = self.fuse(torch.cat([x, edge, context], dim=1))
        gate = self.gate(detail)
        return x + self.alpha * gate * detail

# Module3
class CGF(nn.Module):
    """Channel-Gated Fusion: adaptive channel response selection + cross-fusion

    Implements channel-wise gating via group normalization and adaptive
    thresholding, followed by cross-channel information mixing. Designed
    for fine-grained feature refinement in dual-branch architectures.
    """

    def __init__(self, channels, group_num=4, gate_threshold=0.5):
        super().__init__()
        self.gn = nn.GroupNorm(group_num, channels)
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise response calibration via group normalization
        gx = self.gn(x)

        # Compute adaptive channel weights for gating
        w = self.gn.weight.view(1, -1, 1, 1)
        w = w / (w.abs().sum() + 1e-6)
        gate = self.sigmoid(gx * w)

        # Adaptive feature partitioning based on response magnitude
        info = (gate >= self.gate_threshold).float() * gx
        less = (gate < self.gate_threshold).float() * gx

        # Cross-channel fusion for enhanced information propagation
        c = info.size(1)
        if c % 2 != 0:
            return gx

        i1, i2 = torch.chunk(info, 2, dim=1)
        l1, l2 = torch.chunk(less, 2, dim=1)
        out = torch.cat([i1 + l2, i2 + l1], dim=1)
        return out


class CCA(nn.Module):
    """Compressed Channel Aggregation: hierarchical channel processing with attention

    Implements channel-split compression with hybrid convolution operations
    and attention-based recalibration. Optimized for efficient feature
    aggregation while preserving discriminative channel information.
    """

    def __init__(self, channels, alpha=0.5, squeeze_ratio=2, groups=2):
        super().__init__()
        # Hierarchical channel partitioning
        up_c = int(channels * alpha)
        low_c = channels - up_c
        self.up_c = up_c
        self.low_c = low_c

        # Dimensionality reduction for computational efficiency
        s_up = max(1, up_c // squeeze_ratio)
        s_low = max(1, low_c // squeeze_ratio)

        self.squeeze_up = nn.Conv2d(up_c, s_up, 1, bias=False)
        self.squeeze_low = nn.Conv2d(low_c, s_low, 1, bias=False)

        # Hybrid convolution: group + pointwise for complementary feature mixing
        self.gwc = nn.Conv2d(s_up, channels, 3, 1, 1, groups=groups, bias=False)
        self.pwc_up = nn.Conv2d(s_up, channels, 1, bias=False)
        self.pwc_low = nn.Conv2d(s_low, channels, 1, bias=False)

        # Channel-wise attention for adaptive feature weighting
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Split channels for hierarchical processing
        up, low = torch.split(x, [self.up_c, self.low_c], dim=1)
        up = self.squeeze_up(up)
        low = self.squeeze_low(low)

        # Complementary feature aggregation via hybrid convolutions
        y_up = self.gwc(up) + self.pwc_up(up)
        y_low = self.pwc_low(low)

        # Fuse and apply channel-wise attention
        y = y_up + y_low
        att = self.gate(y)
        return y * att


class FEB(nn.Module):
    """Feature Enhancement Block: collaborative spatial-channel refinement

    Cascades Channel-Gated Fusion (CGF) and Compressed Channel Aggregation
    (CCA) for joint spatial-channel feature enhancement. Serves as the
    core refinement unit in parallel high-frequency architectures.
    """

    def __init__(self, channels, group_num=4, gate_threshold=0.5):
        super().__init__()
        self.cgf = CGF(channels, group_num, gate_threshold)
        self.cca = CCA(channels)

    def forward(self, x):
        # Sequential spatial then channel refinement
        return self.cca(self.cgf(x))


class SPE(nn.Module):
    """Structural Prior Extractor: fixed kernels for geometric pattern extraction

    Employs predefined structural kernels to extract geometric discontinuities
    and spatial patterns. Parameters are frozen to provide stable structural
    priors without increasing training complexity or overfitting risk.
    """

    def __init__(self, channels):
        super().__init__()
        # Predefined structural kernels for orthogonal pattern extraction
        kx = torch.tensor(
            [[-3., 0., 3.],
             [-10., 0., 10.],
             [-3., 0., 3.]]
        )
        ky = torch.tensor(
            [[-3., -10., -3.],
             [0., 0., 0.],
             [3., 10., 3.]]
        )

        # Depthwise convolution for per-channel structural extraction
        self.conv_x = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.conv_y = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)

        # Initialize with fixed structural priors (non-trainable)
        with torch.no_grad():
            self.conv_x.weight.copy_(kx.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
            self.conv_y.weight.copy_(ky.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))

        # Freeze parameters to maintain prior stability
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Extract orthogonal structural responses and compute magnitude
        gx = self.conv_x(x)
        gy = self.conv_y(x)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)


class MDE(nn.Module):
    """Multi-Direction Encoder: angular-aware feature encoding via rotated kernels

    Implements parallel directional encoders with rotated structural kernels
    to capture orientation-sensitive patterns. Designed for anisotropic
    structure awareness in high-frequency enhancement tasks.
    """

    def __init__(self, in_channels, branch_channels, num_directions=4):
        super().__init__()
        # Configurable angular sampling for comprehensive directional coverage
        angles = [0, 45, 90, 135][:num_directions]

        # Feature dimension reduction for efficient multi-branch processing
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.branches = nn.ModuleList()

        # Initialize directional encoders with rotated structural kernels
        for angle in angles:
            conv = nn.Conv2d(branch_channels, branch_channels, 3, 1, 1,
                             groups=branch_channels, bias=False)
            weight = self._rotated_kernel(angle, branch_channels)
            with torch.no_grad():
                conv.weight.copy_(weight)
            self.branches.append(conv)

        # Fusion layer to integrate multi-directional features
        self.fuse = nn.Conv2d(branch_channels * num_directions, in_channels, 1, bias=True)

    def _rotated_kernel(self, angle, channels):
        """Generate rotated structural kernel via coordinate transformation"""
        base = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]]
        )
        theta = math.radians(angle)
        rot = torch.zeros_like(base)
        center = 1

        # Coordinate rotation for kernel orientation adaptation
        for i in range(3):
            for j in range(3):
                x = j - center
                y = i - center
                xr = x * math.cos(theta) - y * math.sin(theta)
                yr = x * math.sin(theta) + y * math.cos(theta)
                xi = int(round(xr)) + center
                yi = int(round(yr)) + center
                if 0 <= xi < 3 and 0 <= yi < 3:
                    rot[i, j] = base[yi, xi]

        return rot.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)

    def forward(self, x):
        # Reduce dimensionality before directional encoding
        x = self.reduce(x)

        # Parallel directional feature extraction with non-linear activation
        outs = [F.leaky_relu(branch(x), 0.1, inplace=True) for branch in self.branches]
        outs = torch.cat(outs, dim=1)

        # Fuse multi-directional features to original channel dimension
        return self.fuse(outs)


class PHASE(nn.Module):
    """Parallel High-frequency Aware Structure Enhancement

    Integrates three complementary pathways for high-frequency structure
    enhancement: (1) collaborative spatial-channel refinement, (2) fixed
    structural prior extraction, and (3) multi-directional geometry encoding.
    Outputs residual detail features for fine-grained texture restoration.

    This module is designed for dual-branch architectures where one branch
    handles resolution restoration and this module provides high-frequency
    detail enhancement through parallel structural cue fusion.

    Args:
        channels (int): Input/output feature channels
        res_scale (float): Residual scaling factor for stable training
    """

    def __init__(self, channels, res_scale=0.1):
        super().__init__()
        # Pre-projection for feature alignment
        self.pre = nn.Conv2d(channels, channels, 1, bias=True)

        # Core feature enhancement block
        self.feb = FEB(channels)

        # Structural prior branch: fixed kernels for stable geometric cues
        self.spe = SPE(channels)

        # Multi-directional branch: adaptive angular feature encoding
        self.mde = MDE(
            in_channels=channels,
            branch_channels=max(8, channels // 4),
            num_directions=4
        )

        # Multi-cue fusion network
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        self.res_scale = res_scale

    def forward(self, x):
        """
        Forward pass for parallel high-frequency structure enhancement.

        Args:
            x (Tensor): Input feature map [B, C, H, W]

        Returns:
            Tensor: Residual detail features scaled by res_scale,
                   intended to be fused with upsampling branch outputs
                   for comprehensive detail restoration.
        """
        # Base feature enhancement with residual connection
        base = self.feb(self.pre(x)) + x

        # Extract complementary structural cues from parallel branches
        struct_prior = self.spe(base)  # Fixed structural patterns
        direction_encoding = self.mde(base)  # Adaptive directional features

        # Fuse multi-cue representations for comprehensive high-frequency enhancement
        detail = self.fuse(torch.cat([base, struct_prior, direction_encoding], dim=1))

        # Scaled residual output for stable gradient flow
        return self.res_scale * detail