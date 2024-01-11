# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypeAlias

from torch import device, dtype

from audioseal.models import AudioSealDectector, AudioSealWM


Device: TypeAlias = device

DataType: TypeAlias = dtype


@dataclass
class SEANetConfig:
    """
    Map common hparams of SEANet encoder and decoder
    """
    channels: int
    dimension: int
    n_filters: int
    n_residual_layers: int
    ratios: List[int]
    activation: str
    activation_params: Dict[str, float]
    norm: Literal["none", "weight_norm", "spectral_norm", "time_group_norm"]
    norm_params: Dict[str, Any]
    kernel_size: int
    last_kernel_size: int
    residual_kernel_size: int
    dilation_base: int
    causal: bool
    pad_mode: str
    true_skip: bool
    compress: int
    lstm: int
    disable_norm_outer_blocks: int


@dataclass
class EncoderConfig:
    ...


@dataclass
class DecoderConfig:
    final_activation: Optional[str]
    final_activation_params: Optional[dict]
    trim_right_ratio: float


@dataclass
class DetectorConfig:
    output_dim: int


@dataclass
class AudioSealWMConfig:
    sample_rate: int
    channels: int
    nbits: int
    seanet: SEANetConfig
    encoder: EncoderConfig
    decoder: DecoderConfig


@dataclass
class AudioSealDetectorConfig:
    sample_rate: int
    channels: int
    nbits: int
    seanet: SEANetConfig
    detector: DetectorConfig



def create_generator(
    config: AudioSealWMConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> AudioSealWM:
    ...


def create_detector(
    config: AudioSealDetectorConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> AudioSealDectector:
    ...

