# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from audioseal.models import AudioSealWM, AudioSealDectector

def load_watermarker(model_name: str) -> AudioSealWM:
    ...


def load_detector(model_name: str) -> AudioSealDectector:
    ...
