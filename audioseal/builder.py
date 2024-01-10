# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional
from typing_extensions import TypeAlias

from torch import device, dtype

from audioseal.models import AudioSealDectector, AudioSealWM


Device: TypeAlias = device

DataType: TypeAlias = dtype


@dataclass
class AudioSealDetectorConfig:
    """
    Hold the configuration for the AudioSealDetector.
    Most of the params below are for the underlying
    SEANetEncoder model
    """
    dummy: int


@dataclass
class AudioSealWMConfig:
    ...


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

