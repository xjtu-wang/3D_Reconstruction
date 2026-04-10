from dataclasses import dataclass
from typing import Any, List, Optional

import torch 
import torch.nn as nn

from .config import ModelConfig

import MinkowskiEngine as ME


def _replace_sparse_features(tensor: Any, features: torch.Tensor) -> Any:
    if hasattr(tensor, "replace_feature"):
        return tensor.replace_feature(features)
    return ME.SparseTensor(
        features=features,
        coordinate_map_key=tensor.coordinate_map_key,
        coordinate_manager=tensor.coordinate_manager,
    )

class SparseConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Any,stride: Any = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dimension=4,
                bias=False,
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
        )

    def forward(self, x:Any) -> Any:
        return self.net(x)
    
class SparseTransposeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Any, stride: Any) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dimension=4,
                bias=False,
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
        )

    def forward(self, x: Any) -> Any:
        return self.net(x)
    
class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, config: ModelConfig) -> None:
        super().__init__()
        self.downsample = SparseConvBlock(
            in_channels,
            out_channels,
            kernel_size=config.stride_kernel,
            stride=config.stride_kernel,
        )
        self.refine = SparseConvBlock(
            out_channels,
            out_channels,
            kernel_size=config.spacetime_kernel,
        )

    def forward(self, x: Any) -> Any:
        x = self.downsample(x)
        x = self.refine(x)
        return x
    
class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, config: ModelConfig) -> None:
        super().__init__()
        self.upsample = SparseTransposeBlock(
            in_channels,
            out_channels,
            kernel_size=config.stride_kernel,
            stride=config.stride_kernel,
        )
        self.fuse = SparseConvBlock(
            out_channels + skip_channels,
            out_channels,
            kernel_size=config.spacetime_kernel,
        )

    def forward(self, x: Any, skip: Any) -> Any:
        x = self.upsample(x)
        skip_features = skip.features_at_coordinates(x.C.float())  # Ensure skip features are aligned with x
        skip_on_x = ME.SparseTensor(
            features=skip_features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        x = ME.cat(x, skip_on_x)
        x = self.fuse(x)
        return x
    
@dataclass
class OccupancyPrediction:
    stride: int
    logits: Any

@dataclass
class ModelOutput:
    offsets : Any
    occupancy_predictions: List[OccupancyPrediction]

class TerrainReconstructionModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.pruning = ME.MinkowskiPruning()

        self.stem = SparseConvBlock(
            self.config.in_channels,
            self.config.stem_channels,
            kernel_size=self.config.spacetime_kernel,
        )

        encoder_channels = self.config.encoder_channels
        decoder_channels = self.config.decoder_channels

        self.enc1 = EncoderStage(self.config.stem_channels, encoder_channels[0], self.config)
        self.enc2 = EncoderStage(encoder_channels[0], encoder_channels[1], self.config)
        self.enc3 = EncoderStage(encoder_channels[1], encoder_channels[2], self.config)
        self.enc4 = EncoderStage(encoder_channels[2], encoder_channels[3], self.config)

        self.dec3 = DecoderStage(encoder_channels[3], encoder_channels[2], decoder_channels[0], self.config)
        self.dec2 = DecoderStage(decoder_channels[0], encoder_channels[1], decoder_channels[1], self.config)
        self.dec1 = DecoderStage(decoder_channels[1], encoder_channels[0], decoder_channels[2], self.config)
        self.dec0 = DecoderStage(decoder_channels[2], self.config.stem_channels, decoder_channels[3], self.config)

        self.occ3 = ME.MinkowskiConvolution(decoder_channels[0], 1, kernel_size=1, dimension=4)
        self.occ2 = ME.MinkowskiConvolution(decoder_channels[1], 1, kernel_size=1, dimension=4)
        self.occ1 = ME.MinkowskiConvolution(decoder_channels[2], 1, kernel_size=1, dimension=4)
        self.occ0 = ME.MinkowskiConvolution(decoder_channels[3], 1, kernel_size=1, dimension=4)
        self.offset_head = ME.MinkowskiConvolution(decoder_channels[3], 3, kernel_size=1, dimension=4)

    def _prune(self, tensor: Any, logits: Any, threshold: float) -> Any:
        if tensor.F.shape[0] == 0:
            return tensor
        keep_mask = torch.sigmoid(logits.F).squeeze(1) >= threshold
        if not torch.any(keep_mask):
            keep_mask = keep_mask.clone()
            keep_mask[torch.argmax(logits.F.squeeze(1))] = True
        return self.pruning(tensor, keep_mask)
    
    def forward(self, x: Any, pruning_threshold: Optional[float] = None) -> ModelOutput:
        threshold = pruning_threshold
        if threshold is None:
            threshold = self.config.pruning_threshold

        stem = self.stem(x)
        enc1 = self.enc1(stem)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        latent = self.enc4(enc3)

        dec3 = self.dec3(latent, enc3)
        occ3 = self.occ3(dec3)
        dec3 = self._prune(dec3, occ3, threshold)

        dec2 = self.dec2(dec3, enc2)
        occ2 = self.occ2(dec2)
        dec2 = self._prune(dec2, occ2, threshold)

        dec1 = self.dec1(dec2, enc1)
        occ1 = self.occ1(dec1)
        dec1 = self._prune(dec1, occ1, threshold)

        dec0 = self.dec0(dec1, stem)
        occ0 = self.occ0(dec0)
        dec0 = self._prune(dec0, occ0, threshold)

        offsets = self.offset_head(dec0)
        offsets = _replace_sparse_features(offsets, torch.sigmoid(offsets.F))

        return ModelOutput(
            offsets=offsets,
            occupancy_predictions=[
                OccupancyPrediction(stride=8, logits=occ3),
                OccupancyPrediction(stride=4, logits=occ2),
                OccupancyPrediction(stride=2, logits=occ1),
                OccupancyPrediction(stride=1, logits=occ0),
            ],
        )
