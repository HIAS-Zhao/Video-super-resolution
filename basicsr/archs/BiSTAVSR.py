import torch
from typing import List, Optional
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.arch_util import flow_warp
from basicsr.archs.basicvsr_arch import (
    ConvResidualBlocks,
    EDVRFeatureExtractor,
    KBAFunction,
    SimpleGate,
)
from basicsr.archs.channel_diversity import MultiSpectralAttentionLayer
from basicsr.archs.spynet_arch import SpyNet
from basicsr.utils.registry import ARCH_REGISTRY

from .function import BiSTA, OAR, PHASE


def SDE(x, att, kernel_size, groups, bias_bank, weight_bank):
    return KBAFunction.apply(x, att, kernel_size, groups, bias_bank, weight_bank)


class MADE(nn.Module):
    """Multi-Axis Diversity Enhancement retained from the MADNet baseline."""

    def __init__(self, c=64, DW_Expand=2, FFN_Expand=2, nset=64, k=3, gc=4, lightweight=False):
        super().__init__()
        self.k = k
        self.g = c // gc
        self.w = nn.Parameter(torch.zeros(1, nset, c * c // self.g * k**2))
        self.b = nn.Parameter(torch.zeros(1, nset, c))

        self.dwconv_k5 = nn.Conv2d(c, c, 5, 1, 2, bias=True)
        self.channel_diversity = MultiSpectralAttentionLayer(
            channel=c, dct_w=56, dct_h=56)
        self.conv1 = nn.Conv2d(c, c, 1, bias=True)
        self.conv21 = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=True)
        self.conv11 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=True),
            nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, 32, 3, 1, 1, groups=32, bias=True),
            SimpleGate(),
            nn.Conv2d(16, 64, 1),
        )
        self.conv211 = nn.Conv2d(64, 64, 1)
        self.attgamma = nn.Parameter(torch.full((1, 64, 1, 1), 1e-2))
        self.ga1 = nn.Parameter(torch.full((1, c, 1, 1), 1e-2))
        self.fusion = nn.Conv2d(c, c, 1, bias=True)

    def forward(self, inp):
        x = self.dwconv_k5(inp)
        channel_feature = self.channel_diversity(x)
        auxiliary_feature = self.conv11(x)

        attention = self.conv2(x) * self.attgamma + self.conv211(x)
        unfolded_feature = self.conv21(self.conv1(x))
        spatial_feature = SDE(
            unfolded_feature,
            attention,
            self.k,
            self.g,
            self.b,
            self.w,
        ) * self.ga1 + unfolded_feature

        return self.fusion(channel_feature + auxiliary_feature + spatial_feature) * inp

@ARCH_REGISTRY.register()
class BiSTAVSR(nn.Module):
    """Offset-Aware Bidirectional Spatio-Temporal Aggregation for VSR.

    A video super-resolution architecture that integrates:
    1. Keyframe feature extraction via EDVR-based pyramid alignment
    2. Bidirectional temporal propagation with flow-guided warping
    3. Offset-Aware Refinement (OAR) for structure-guided matching
    4. Bidirectional Spatio-Temporal Aggregator (BiSTA) for symmetric
       neighborhood feature fusion
    5. Parallel High-frequency Aware Structure Enhancement (PHASE) for
       fine-grained detail restoration

    This design enables accurate temporal alignment and high-frequency
    detail recovery while maintaining computational efficiency through
       parameter sharing and residual learning.

    Args:
        num_feat (int): Number of feature channels. Default: 64.
        num_block (int): Number of residual blocks for propagation trunk. Default: 15.
        keyframe_stride (int): Stride for keyframe selection. Default: 5.
        temporal_padding (int): Number of frames padded for keyframe extraction. Default: 2.
        dropout (float): Dropout rate for attention modules. Default: 0.0.
        spynet_path (str, optional): Path to pretrained SPyNet weights.
        edvr_path (str, optional): Path to pretrained EDVR weights.

    Input:
        x (Tensor): Input video sequence (B, T, 3, H, W)

    Output:
        Tensor: Reconstructed HR sequence (B, T, 3, 4H, 4W)
    """

    def __init__(self,
                 num_feat: int = 64,
                 num_block: int = 15,
                 keyframe_stride: int = 5,
                 temporal_padding: int = 2,
                 dropout: float = 0.0,
                 spynet_path: Optional[str] = None,
                 edvr_path: Optional[str] = None):
        super().__init__()

        self.num_feat = num_feat
        self.temporal_padding = temporal_padding
        self.keyframe_stride = keyframe_stride

        # ===== Keyframe feature extraction =====
        self.edvr = EDVRFeatureExtractor(
            num_input_frame=temporal_padding * 2 + 1,
            num_feat=num_feat,
            load_path=edvr_path
        )

        # ===== Optical flow estimation for alignment =====
        self.spynet = SpyNet(spynet_path)

        # ===== Bidirectional spatio-temporal aggregation =====
        # Replaces StackedSpatialTemporalTransformer with BiSTA
        self.temporal_agg = BiSTA(
            feat_channels=num_feat,
            encoding_dim=64,
            num_heads=8,
            mlp_ratio=2,
            dropout=dropout,
            add_ref_residual=False
        )

        # ===== Backward propagation branch =====
        self.back_attn_conv = ConvResidualBlocks(num_feat * 3, num_feat * 2, 3)
        self.backward_conv = ConvResidualBlocks(3, num_feat, 3)
        self.backward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.backward_trunk = ConvResidualBlocks(num_feat * 3, num_feat, num_block)

        # ===== Forward propagation branch =====
        self.forward_attn_conv = ConvResidualBlocks(num_feat * 3, num_feat * 2, 3)
        self.forward_conv = ConvResidualBlocks(3, num_feat, 3)
        self.forward_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1, bias=True)
        self.forward_trunk = ConvResidualBlocks(num_feat * 4, num_feat, num_block)

        # ===== Reconstruction head (main branch) =====
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

        # ===== Detail enhancement branch (parallel) =====
        # Replaces VSRDetailEnhance with PHASE
        self.upconv1_s = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2_s = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.detail_enhance = PHASE(channels=64)
        self.conv_hr_s = nn.Conv2d(64, 64, 3, 1, 1)
        self.detail_alpha = nn.Parameter(torch.tensor(0.05))
        self.conv_last_s = nn.Conv2d(64, 3, 3, 1, 1)

        # ===== Activation =====
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # ===== Offset-Aware Refinement for attention guidance =====
        # Replaces AttentionEGA with OAR
        self.offset_guidance = OAR(num_feat)
        self.made = MADE()  # Multi-Axis Diversity Enhancement

    @staticmethod
    def convert_state_dict(state_dict):
        """Map training-code names to the paper-aligned module names.

        The released architecture uses BiSTA/OAR/PHASE terminology, while the
        experiment code used ``attention``, ``attn_ega`` and the original
        detail-branch submodule names. This conversion keeps the trained
        checkpoints loadable with ``strict=True``.
        """
        prefix_map = (
            ('attention.da_layer1.', 'temporal_agg.aggregator.'),
            ('attn_ega.', 'offset_guidance.'),
            ('detail_enhance.sc.sru.', 'detail_enhance.feb.cgf.'),
            ('detail_enhance.sc.cru.', 'detail_enhance.feb.cca.'),
            ('detail_enhance.scharr.', 'detail_enhance.spe.'),
            ('detail_enhance.grad.', 'detail_enhance.mde.'),
        )

        converted = state_dict.__class__()
        if hasattr(state_dict, '_metadata'):
            converted._metadata = state_dict._metadata
        for key, value in state_dict.items():
            new_key = key
            for old_prefix, new_prefix in prefix_map:
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix):]
                    break
            converted[new_key] = value
        return converted

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict = self.convert_state_dict(state_dict)
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except TypeError:
            # Compatibility with PyTorch versions before the ``assign`` flag.
            return super().load_state_dict(state_dict, strict=strict)

    def pad_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Pad input to ensure resolution is divisible by 4 (for EDVR compatibility)."""
        n, t, c, h, w = x.size()
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        x = x.view(-1, c, h, w)
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        return x.view(n, t, c, h + pad_h, w + pad_w)

    def get_flow(self, x: torch.Tensor):
        """Compute bidirectional optical flows between consecutive frames."""
        b, n, c, h, w = x.size()
        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)
        return flows_forward, flows_backward

    def get_keyframe_feature(self, x: torch.Tensor, keyframe_idx: List[int]):
        """Extract pyramid-aligned features for keyframes using EDVR."""
        if self.temporal_padding == 2:
            x = [x[:, [4, 3]], x, x[:, [-4, -5]]]
        elif self.temporal_padding == 3:
            x = [x[:, [6, 5, 4]], x, x[:, [-5, -6, -7]]]
        x = torch.cat(x, dim=1)
        num_frames = 2 * self.temporal_padding + 1
        feats_keyframe = {}
        for i in keyframe_idx:
            feats_keyframe[i] = self.edvr(x[:, i:i + num_frames].contiguous())
        return feats_keyframe

    def _compose_flow(self, flow_i_to_j: torch.Tensor, flow_j_to_k: torch.Tensor) -> torch.Tensor:
        """Compose flows: f_{i→k}(p) = f_{i→j}(p) + f_{j→k}(p + f_{i→j}(p))"""
        warped = flow_warp(flow_j_to_k, flow_i_to_j.permute(0, 2, 3, 1))
        return flow_i_to_j + warped

    def _collect_aligned_neighbors(
            self,
            center_idx: int,
            feat_all: torch.Tensor,
            flows_forward: torch.Tensor,
            flows_backward: torch.Tensor,
            num_frames: int
    ):
        """Collect and align neighboring frames to current frame (max 2 left + 2 right).

        Returns:
            supp_feats: List of aligned support features [left...right]
            left_len: Number of frames in left neighborhood
        """
        supp_feats = []
        left_len = 0

        # Left neighborhood: i-2, i-1 (past frames)
        if center_idx - 2 >= 0:
            flow_i_to_i1 = flows_forward[:, center_idx - 1]
            flow_i1_to_i2 = flows_forward[:, center_idx - 2]
            flow_i_to_i2 = self._compose_flow(flow_i_to_i1, flow_i1_to_i2)
            f = feat_all[:, center_idx - 2]
            f = flow_warp(f, flow_i_to_i2.permute(0, 2, 3, 1))
            supp_feats.append(f)
            left_len += 1
        if center_idx - 1 >= 0:
            f = feat_all[:, center_idx - 1]
            f = flow_warp(f, flows_forward[:, center_idx - 1].permute(0, 2, 3, 1))
            supp_feats.append(f)
            left_len += 1

        # Right neighborhood: i+1, i+2 (future frames)
        if center_idx + 1 < num_frames:
            f = feat_all[:, center_idx + 1]
            f = flow_warp(f, flows_backward[:, center_idx].permute(0, 2, 3, 1))
            supp_feats.append(f)
        if center_idx + 2 < num_frames:
            flow_i_to_i1 = flows_backward[:, center_idx]
            flow_i1_to_i2 = flows_backward[:, center_idx + 1]
            flow_i_to_i2 = self._compose_flow(flow_i_to_i1, flow_i1_to_i2)
            f = feat_all[:, center_idx + 2]
            f = flow_warp(f, flow_i_to_i2.permute(0, 2, 3, 1))
            supp_feats.append(f)

        return supp_feats, left_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for video super-resolution.

        Args:
            x: Input LR video sequence (B, T, 3, H, W)

        Returns:
            Reconstructed HR sequence (B, T, 3, 4H, 4W)
        """
        b, n, _, h_input, w_input = x.size()
        x = self.pad_spatial(x)
        h, w = x.shape[3:]

        # Select keyframes for pyramid feature extraction
        keyframe_idx = list(range(0, n, self.keyframe_stride))
        if keyframe_idx[-1] != n - 1:
            keyframe_idx.append(n - 1)

        # Compute flows and keyframe features
        flows_forward, flows_backward = self.get_flow(x)
        feats_keyframe = self.get_keyframe_feature(x, keyframe_idx)

        # Pre-compute per-frame features for efficiency
        # - feat_all_raw: base features for value propagation
        # - feat_all_attn: OAR-enhanced features for attention Q/K
        # - feat_all: MADE-enhanced features for main propagation
        x_flat = x.view(-1, 3, h, w)
        feat_all_raw = self.backward_conv(x_flat)
        feat_all_attn = self.offset_guidance(feat_all_raw)  # OAR for offset-aware refinement
        feat_all = self.made(feat_all_raw)
        feat_all_raw = feat_all_raw.view(b, n, self.num_feat, h, w)
        feat_all_attn = feat_all_attn.view(b, n, self.num_feat, h, w)
        feat_all = feat_all.view(b, n, self.num_feat, h, w)

        # ===== Backward propagation (right-to-left) =====
        out_l = []
        attn_feats = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)

        for i in range(n - 1, -1, -1):
            x_f_attn = feat_all_attn[:, i]  # OAR-enhanced for attention
            x_f = feat_all[:, i]  # MADE-enhanced for propagation
            attn_feat = x_f_attn.new_zeros(b, self.num_feat * 2, h, w)

            # Bidirectional temporal aggregation via BiSTA
            supp_feats, left_len = self._collect_aligned_neighbors(
                i, feat_all_attn, flows_forward, flows_backward, n
            )
            supp_value_feats, _ = self._collect_aligned_neighbors(
                i, feat_all_raw, flows_forward, flows_backward, n
            )
            if len(supp_feats) > 0:
                attn_feat = self.temporal_agg(  # BiSTA aggregation
                    x_f_attn, supp_feats, left_len=left_len, supp_value_feats=supp_value_feats
                )

            # Flow-guided propagation from next frame
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            # Fuse keyframe features at keyframe positions
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
                feat_prop = self.backward_fusion(feat_prop)

            # Integrate aggregated neighborhood features
            x_f = self.back_attn_conv(torch.cat([x_f, attn_feat], dim=1))
            feat_prop = torch.cat([x_f, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)

            out_l.insert(0, feat_prop)
            attn_feats.insert(0, attn_feat)

        # ===== Forward propagation (left-to-right) =====
        feat_prop = torch.zeros_like(feat_prop)

        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            x_f = self.made(self.forward_conv(x_i))

            # Flow-guided propagation from previous frame
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            # Fuse keyframe features
            if i in keyframe_idx:
                feat_prop = torch.cat([feat_prop, feats_keyframe[i]], dim=1)
                feat_prop = self.forward_fusion(feat_prop)

            # Integrate backward-aggregated features
            x_f = self.forward_attn_conv(torch.cat([x_f, attn_feats[i]], dim=1))
            feat_prop = torch.cat([x_f, out_l[i], feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # ===== Main reconstruction branch =====
            out = self.lrelu(self.pixel_shuffle(self.upconv1(feat_prop)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            hr_feat = self.lrelu(self.conv_hr(out))
            out = self.conv_last(hr_feat)

            # ===== Parallel detail enhancement branch (PHASE) =====
            detail_residual = self.detail_enhance(feat_prop)  # PHASE module
            detail_residual = self.lrelu(self.pixel_shuffle(self.upconv1_s(detail_residual)))
            detail_residual = self.lrelu(self.pixel_shuffle(self.upconv2_s(detail_residual)))
            hr_feat_s = self.lrelu(self.conv_hr_s(detail_residual))
            detail = self.conv_last_s(hr_feat_s)

            # Fuse main output + detail enhancement + bicubic baseline
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out = out + self.detail_alpha * detail + base
            out_l[i] = out

        # Crop to original padded size and stack temporal outputs
        return torch.stack(out_l, dim=1)[..., :4 * h_input, :4 * w_input]
