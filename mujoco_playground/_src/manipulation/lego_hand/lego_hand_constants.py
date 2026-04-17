# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Constants for lego hand."""

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "manipulation" / "lego_hand"
CUBE_XML = ROOT_PATH / "xmls" / "scene.xml"
DEX_CUBE_XML = ROOT_PATH / "xmls" / "scene_dex_cube.xml"

NQ = 20
NV = 20
NU = 20

JOINT_NAMES = [
    # finger 1
    "f1_finger_palm_joint",
    "f1_finger_base_joint",
    "f1_finger_middle_joint",
    "f1_finger_end_joint",
    # finger 2
    "f2_finger_palm_joint",
    "f2_finger_base_joint",
    "f2_finger_middle_joint",
    "f2_finger_end_joint",
    # finger 3
    "f3_finger_palm_joint",
    "f3_finger_base_joint",
    "f3_finger_middle_joint",
    "f3_finger_end_joint",
    # finger 4
    "f4_finger_palm_joint",
    "f4_finger_base_joint",
    "f4_finger_middle_joint",
    "f4_finger_end_joint",
    # finger 5
    "f5_finger_palm_joint",
    "f5_finger_base_joint",
    "f5_finger_middle_joint",
    "f5_finger_end_joint",
]

ACTUATOR_NAMES = [
    # finger 1
    "f1_finger_palm_joint",
    "f1_finger_base_joint",
    "f1_finger_middle_joint",
    "f1_finger_end_joint",
    # finger 2
    "f2_finger_palm_joint",
    "f2_finger_base_joint",
    "f2_finger_middle_joint",
    "f2_finger_end_joint",
    # finger 3
    "f3_finger_palm_joint",
    "f3_finger_base_joint",
    "f3_finger_middle_joint",
    "f3_finger_end_joint",
    # finger 4
    "f4_finger_palm_joint",
    "f4_finger_base_joint",
    "f4_finger_middle_joint",
    "f4_finger_end_joint",
    # finger 5
    "f5_finger_palm_joint",
    "f5_finger_base_joint",
    "f5_finger_middle_joint",
    "f5_finger_end_joint",
]

FINGERTIP_NAMES = [
    "f1_finger_end_lever",
    "f2_finger_end_lever",
    "f3_finger_end_lever",
    "f4_finger_end_lever",
    "f5_finger_end_lever",
]
