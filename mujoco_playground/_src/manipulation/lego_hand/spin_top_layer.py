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
"""Spin top layer task for lego hand."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.manipulation.lego_hand import base as lego_hand_base
from mujoco_playground._src.manipulation.lego_hand import lego_hand_constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.01,
      action_scale=0.5,
      action_repeat=1,
      ema_alpha=1.0,
      episode_length=1000,
      success_threshold=0.02,
      history_len=1,
      obs_noise=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.05,
              cube_pos=0.02,
              cube_ori=0.1,
          ),
          random_ori_injection_prob=0.0,
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              step=-0.0,
              progress=150.0,
              ori=75.0,
              hold=30.0,
              pos=-20.0,
              core_ori=-10.0,
              act_rate=-0.5,
              vel=-0.5,
              energy=-0.0,
              reach=10.0,
          ),
          success_reward=1000.0,
          term_penalty=-20.0,
          hold_pos_margin=0.02,
          hold_ori_margin=jp.pi / 6.0,
          hold_vel_margin=0.1,
          core_ori_margin=jp.pi / 6.0,
          reach_margin_grad=0.05,
          ori_margin_inner=0.02,
          ori_margin_outer=jp.pi,
          pos_margin=0.015,
          v_max=5.0,
          v_margin=2.0,
          act_rate_max=80.0,
          vel_excess_max=26.0,
          energy_max=5000.0,
      ),
      pert_config=config_dict.create(
          enable=False,
          linear_velocity_pert=[0.0, 3.0],
          angular_velocity_pert=[0.0, 0.5],
          pert_duration_steps=[1, 100],
          pert_wait_steps=[60, 150],
      ),
      impl='warp',
      naconmax=30 * 8192,
      naccdmax=200,
      njmax=500,
  )


class SpinTopLayer(lego_hand_base.LegoHandEnv):
  """Spin the top layer of a rubiks cube by 90 degrees."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.CUBE_XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    home_key = self._mj_model.keyframe("home")
    self._init_q = jp.array(home_key.qpos, dtype=float)
    self._init_mpos = jp.array(home_key.mpos, dtype=float).reshape(-1, 3)
    self._init_mquat = jp.array(home_key.mquat, dtype=float).reshape(-1, 4)
    self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
    self._uppers = self._mj_model.actuator_ctrlrange[:, 1]
    
    self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
    self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
    self._cube_qids = mjx_env.get_qpos_ids(self.mj_model, ["cube_freejoint"])
    self._cube_dqids = mjx_env.get_qvel_ids(self.mj_model, ["cube_freejoint"])
    
    # Face joints
    self._face_names = ["pZ", "nZ", "pX", "nX", "pY", "nY"]
    self._face_qids = mjx_env.get_qpos_ids(self.mj_model, self._face_names)
    self._face_dqids = mjx_env.get_qvel_ids(self.mj_model, self._face_names)
    self._top_layer_qid = self._face_qids[0]
    self._top_layer_dqid = self._face_dqids[0]

    self._floor_geom_id = self._mj_model.geom("floor").id
    self._cube_geom_id = self._mj_model.geom("cube_collision").id
    self._cube_body_id = self._mj_model.body("core").id
    self._cube_mass = self._mj_model.body_subtreemass[self._cube_body_id]
    self._default_pose = self._init_q[self._hand_qids]
    self._cube_origin = jp.array([0.04, 0.001, 0.0666]) # PALM_CENTER
    
    self._target_face_angle = jp.pi / 2.0

    # Face body IDs for termination distance check
    self._face_body_ids = [self._mj_model.body(n).id for n in self._face_names]

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, pos_rng, vel_rng = jax.random.split(rng, 3)
    q_hand = jp.clip(
        self._default_pose + 0.1 * jax.random.normal(pos_rng, (consts.NQ,)),
        self._lowers,
        self._uppers,
    )
    v_hand = 0.0 * jax.random.normal(vel_rng, (consts.NV,))

    rng, p_rng = jax.random.split(rng, 2)
    start_pos = self._cube_origin + jax.random.uniform(
        p_rng, (3,), minval=-0.01, maxval=0.01
    )
    start_quat = jp.array([1.0, 0.0, 0.0, 0.0])
    q_cube = jp.array([*start_pos, *start_quat])
    v_cube = jp.zeros(6)

    q_faces = jp.zeros(6)
    v_faces = jp.zeros(6)

    qpos = self._init_q.at[self._hand_qids].set(q_hand).at[self._cube_qids].set(q_cube).at[self._face_qids].set(q_faces)
    qvel = jp.zeros(self._mj_model.nv).at[self._hand_dqids].set(v_hand).at[self._cube_dqids].set(v_cube).at[self._face_dqids].set(v_faces)
    
    data = mjx_env.make_data(
        self._mj_model,
        qpos=qpos,
        ctrl=q_hand,
        qvel=qvel,
        mocap_pos=self._init_mpos.at[0].set(start_pos),
        mocap_quat=self._init_mquat,
        impl=self._mjx_model.impl.value,
        naconmax=self._config.naconmax,
        naccdmax=self._config.naccdmax,
        njmax=self._config.njmax,
    )

    rng, pert1, pert2, pert3 = jax.random.split(rng, 4)
    pert_wait_steps = jax.random.randint(
        pert1, (1,), minval=self._config.pert_config.pert_wait_steps[0], maxval=self._config.pert_config.pert_wait_steps[1]
    )
    pert_duration_steps = jax.random.randint(
        pert2, (1,), minval=self._config.pert_config.pert_duration_steps[0], maxval=self._config.pert_config.pert_duration_steps[1]
    )
    pert_lin = jax.random.uniform(
        pert3, minval=self._config.pert_config.linear_velocity_pert[0], maxval=self._config.pert_config.linear_velocity_pert[1]
    )
    pert_ang = jax.random.uniform(
        pert3, minval=self._config.pert_config.angular_velocity_pert[0], maxval=self._config.pert_config.angular_velocity_pert[1]
    )
    pert_velocity = jp.array([pert_lin] * 3 + [pert_ang] * 3)

    info = {
        "rng": rng,
        "step": 0,
        "steps_since_last_success": 0,
        "success_count": 0,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": data.ctrl,
        "qpos_error_history": jp.zeros(self._config.history_len * consts.NQ),
        "cube_pos_error_history": jp.zeros(self._config.history_len * 3),
        "cube_ori_error_history": jp.zeros(self._config.history_len * 6),
        "theta_prev": jp.pi / 2.0,
        "init_core_quat": start_quat,
        "pert_wait_steps": pert_wait_steps,
        "pert_duration_steps": pert_duration_steps,
        "pert_vel": pert_velocity,
        "pert_dir": jp.zeros(6, dtype=float),
        "last_pert_step": jp.array([-jp.inf], dtype=float),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["reward/success"] = jp.zeros((), dtype=float)
    metrics["steps_since_last_success"] = 0
    metrics["success_count"] = 0

    obs = self._get_obs(data, info)
    reward_val, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward_val, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state, state.info["rng"])

    delta = action * self._config.action_scale
    motor_targets = state.data.ctrl + delta
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
    motor_targets = (
        self._config.ema_alpha * motor_targets
        + (1 - self._config.ema_alpha) * state.info["motor_targets"]
    )

    data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
    state.info["motor_targets"] = motor_targets

    # Top layer error
    theta_current = data.qpos[self._top_layer_qid]
    q_current = jp.array([jp.cos(theta_current / 2.0), 0.0, 0.0, jp.sin(theta_current / 2.0)])
    q_target = jp.array([jp.cos(self._target_face_angle / 2.0), 0.0, 0.0, jp.sin(self._target_face_angle / 2.0)])
    q_diff = math.quat_mul(math.quat_inv(q_target), q_current)
    vec_norm = jp.linalg.norm(q_diff[1:])
    e_ori = 2.0 * jp.arcsin(jp.clip(vec_norm, a_min=0.0, a_max=1.0))

    success = e_ori < self._config.success_threshold
    state.info["steps_since_last_success"] = jp.where(success, 0, state.info["steps_since_last_success"] + 1)
    state.info["success_count"] = jp.where(success, state.info["success_count"] + 1, state.info["success_count"])
    state.metrics["steps_since_last_success"] = state.info["steps_since_last_success"]
    state.metrics["success_count"] = state.info["success_count"]

    done = self._get_termination(data, state.info)
    obs = self._get_obs(data, state.info)
    
    rewards, e_ori_ret = self._get_reward(data, action, state.info, state.metrics, done, e_ori)
    
    state.info["theta_prev"] = jp.minimum(state.info["theta_prev"], e_ori_ret)

    rewards_scaled = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
    reward_val = sum(rewards_scaled.values()) * self.dt

    reward_val += done * self._config.reward_config.term_penalty
    state.metrics["reward/success"] = success.astype(float)
    reward_val += success * self._config.reward_config.success_reward

    state.info["step"] += 1
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    for k, v in rewards_scaled.items():
      state.metrics[f"reward/{k}"] = v

    done = jp.logical_or(done, success).astype(reward_val.dtype)
    state = state.replace(data=data, obs=obs, reward=reward_val, done=done)
    return state

  def _get_termination(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    core_pos = self.get_cube_position(data)
    drop_termination = core_pos[2] < 0.03
    
    face_positions = jp.array([data.xpos[bid] for bid in self._face_body_ids])
    dists = jp.linalg.norm(face_positions - core_pos, axis=1)
    max_dist = jp.max(dists)
    broken = max_dist > 0.10

    nans = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))
    return drop_termination | broken | nans

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation:
    hand_qpos = data.qpos[self._hand_qids]
    hand_qvel = data.qvel[self._hand_dqids]
    face_qpos = data.qpos[self._face_qids]
    face_qvel = data.qvel[self._face_dqids]
    
    cube_pos = self.get_cube_position(data)
    cube_pos_rel = cube_pos - self._cube_origin
    cube_quat = self.get_cube_orientation(data)
    cube_linvel = self.get_cube_linvel(data)
    cube_angvel = self.get_cube_angvel(data)

    state = jp.concatenate([
        hand_qpos,
        hand_qvel,
        face_qpos,
        face_qvel,
        cube_pos_rel,
        cube_quat,
        cube_linvel,
        cube_angvel,
    ])
    
    return {
        "state": state,
        "privileged_state": state,
    }

  def _get_reward(self, data: mjx.Data, action: jax.Array, info: dict[str, Any], metrics: dict[str, Any], done: jax.Array, e_ori: jax.Array):
    cfg = self._config.reward_config
    
    core_pos = self.get_cube_position(data)
    e_pos = jp.linalg.norm(self._cube_origin - core_pos)
    r_hold_pos = reward.tolerance(e_pos, (0.0, 0.005), margin=cfg.hold_pos_margin, sigmoid="linear")

    core_quat = self.get_cube_orientation(data)
    q_diff_core = math.quat_mul(math.quat_inv(info["init_core_quat"]), core_quat)
    core_vec_norm = jp.linalg.norm(q_diff_core[1:])
    e_core_ori = 2.0 * jp.arcsin(jp.clip(core_vec_norm, 0.0, 1.0))
    r_hold_ori = reward.tolerance(e_core_ori, (0.0, 0.05), margin=cfg.hold_ori_margin, sigmoid="linear")

    v_cube = jp.linalg.norm(self.get_cube_linvel(data))
    r_hold_vel = reward.tolerance(v_cube, (0.0, 0.01), margin=cfg.hold_vel_margin, sigmoid="linear")

    r_hold = r_hold_pos * r_hold_ori * r_hold_vel
    grasp_gate = r_hold

    delta = info["theta_prev"] - e_ori
    progress_rate = jp.maximum(0.0, delta) / self.dt
    r_progress = jp.clip(progress_rate / 5.0, 0.0, 1.0)
    r_ori = reward.tolerance(e_ori, (0.0, cfg.ori_margin_inner), margin=cfg.ori_margin_outer, sigmoid="linear")

    r_pos = 1.0 - reward.tolerance(e_pos, (0.0, 0.0), margin=cfg.pos_margin, sigmoid="linear")
    r_core_ori = 1.0 - reward.tolerance(e_core_ori, (0.0, 0.05), margin=cfg.core_ori_margin, sigmoid="linear")
    
    d1 = action - info["last_act"]
    d2 = action - 2.0 * info["last_act"] + info["last_last_act"]
    act_rate_val = jp.sum(jp.square(d1)) + jp.sum(jp.square(d2))
    r_act_rate = jp.clip(act_rate_val / cfg.act_rate_max, 0.0, 1.0)
    
    all_vels = jp.concatenate([data.qvel[self._face_dqids], data.qvel[self._hand_dqids]])
    v_thresh = cfg.v_max - cfg.v_margin
    excess = jp.maximum(0.0, jp.abs(all_vels) - v_thresh) / cfg.v_margin
    r_vel = jp.clip(jp.sum(excess) / cfg.vel_excess_max, 0.0, 1.0)

    hand_qvel = data.qvel[self._hand_dqids]
    # Use actuator force specifically for the hand actuators
    act_force = data.actuator_force[:consts.NU]
    energy_val = jp.sum(jp.abs(hand_qvel) * jp.abs(act_force))
    r_energy = jp.clip(energy_val / cfg.energy_max, 0.0, 1.0)

    fingertip_pos = self.get_fingertip_positions(data).reshape(-1, 3)
    top_face_pos = data.xpos[self._mj_model.body("pZ").id]
    dist_to_top = jp.linalg.norm(fingertip_pos - top_face_pos, axis=1)
    dist_to_core = jp.linalg.norm(fingertip_pos - core_pos, axis=1)
    
    idx_sorted = jp.argsort(dist_to_top)
    closest_top_idx = idx_sorted[:2]
    other_idx = idx_sorted[2:]

    mean_dist = (jp.sum(dist_to_top[closest_top_idx]) + jp.sum(dist_to_core[other_idx])) / 5.0
    r_reach = reward.tolerance(mean_dist, (0.025, 0.035), margin=cfg.reach_margin_grad, sigmoid="linear")

    task_multiplier = jp.clip(r_progress + r_ori, 0.0, 1.0)
    
    rewards = {
        "step": 1.0,
        "progress": r_progress * grasp_gate,
        "ori": r_ori * grasp_gate,
        "hold": r_hold * task_multiplier,
        "pos": r_pos,
        "core_ori": r_core_ori,
        "act_rate": r_act_rate,
        "vel": r_vel,
        "energy": r_energy,
        "reach": r_reach,
    }
    
    return rewards, e_ori

  def _maybe_apply_perturbation(
      self, state: mjx_env.State, rng: jax.Array
  ) -> mjx_env.State:
    def gen_dir(rng: jax.Array) -> jax.Array:
      directory = jax.random.normal(rng, (6,))
      return directory / jp.linalg.norm(directory)

    def get_xfrc(
        state: mjx_env.State, pert_dir: jax.Array, i: jax.Array
    ) -> jax.Array:
      u_t = 0.5 * jp.sin(jp.pi * i / state.info["pert_duration_steps"])
      force = (
          u_t
          * self._cube_mass
          * state.info["pert_vel"]
          / (state.info["pert_duration_steps"] * self.dt)
      )
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      xfrc_applied = xfrc_applied.at[self._cube_body_id].set(force * pert_dir)
      return xfrc_applied

    step, last_pert_step = state.info["step"], state.info["last_pert_step"]
    start_pert = jp.mod(step, state.info["pert_wait_steps"]) == 0
    start_pert &= step != 0
    last_pert_step = jp.where(start_pert, step, last_pert_step)
    duration = jp.clip(step - last_pert_step, 0, 100_000)
    in_pert_interval = duration < state.info["pert_duration_steps"]

    pert_dir = jp.where(start_pert, gen_dir(rng), state.info["pert_dir"])
    xfrc = get_xfrc(state, pert_dir, duration) * in_pert_interval

    state.info["pert_dir"] = pert_dir
    state.info["last_pert_step"] = last_pert_step
    data = state.data.replace(xfrc_applied=xfrc)
    return state.replace(data=data)
