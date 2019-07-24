# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np

from planet import control
from planet import tools


Task = collections.namedtuple(
    'Task', 'name, env_ctor, max_length, state_components')


def dummy(config, params):
  action_repeat = params.get('action_repeat', 1)
  max_length = 1000 // action_repeat
  state_components = ['reward']
  env_ctor = lambda: control.wrappers.ActionRepeat(
      control.DummyEnv, action_repeat)
  return Task('dummy', env_ctor, max_length, state_components)


def cartpole_balance(config, params):
  action_repeat = params.get('action_repeat', 8)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'cartpole', 'balance',
      params)
  return Task('cartpole_balance', env_ctor, max_length, state_components)


def cartpole_swingup(config, params):
  action_repeat = params.get('action_repeat', 8)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'cartpole', 'swingup',
      params)
  return Task('cartpole_swingup', env_ctor, max_length, state_components)


def finger_spin(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity', 'touch']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'finger', 'spin', params)
  return Task('finger_spin', env_ctor, max_length, state_components)


def cheetah_run(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'cheetah', 'run', params)
  return Task('cheetah_run', env_ctor, max_length, state_components)


def cup_catch(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'ball_in_cup', 'catch',
      params)
  return Task('cup_catch', env_ctor, max_length, state_components)


def walker_walk(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'height', 'orientations', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'walker', 'walk', params)
  return Task('walker_walk', env_ctor, max_length, state_components)


def reacher_easy(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity', 'to_target']
  env_ctor = tools.bind(
      _dm_control_env, action_repeat, max_length, 'reacher', 'easy', params)
  return Task('reacher_easy', env_ctor, max_length, state_components)


def gym_cheetah(config, params):
  # Works with `isolate_envs: process`.
  action_repeat = params.get('action_repeat', 1)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'state']
  env_ctor = tools.bind(
      _gym_env, action_repeat, config.batch_shape[1], max_length,
      'HalfCheetah-v3')
  return Task('gym_cheetah', env_ctor, max_length, state_components)


def gym_racecar(config, params):
  # Works with `isolate_envs: thread`.
  action_repeat = params.get('action_repeat', 1)
  max_length = 1000 // action_repeat
  state_components = ['reward']
  env_ctor = tools.bind(
      _gym_env, action_repeat, config.batch_shape[1], max_length,
      'CarRacing-v0', obs_is_image=True)
  return Task('gym_racing', env_ctor, max_length, state_components)


def _dm_control_env(
    action_repeat, max_length, domain, task, params, normalize=False,
    camera_id=None):
  if isinstance(domain, str):
    from dm_control import suite
    env = suite.load(domain, task)
  else:
    assert task is None
    env = domain()
  if camera_id is None:
    camera_id = int(params.get('camera_id', 0))
  env = control.wrappers.DeepMindWrapper(env, (64, 64), camera_id=camera_id)
  env = control.wrappers.ActionRepeat(env, action_repeat)
  if normalize:
    env = control.wrappers.NormalizeActions(env)
  env = control.wrappers.MaximumDuration(env, max_length)
  env = control.wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
  env = control.wrappers.ConvertTo32Bit(env)
  return env


def _gym_env(action_repeat, min_length, max_length, name, obs_is_image=False):
  import gym
  env = gym.make(name)
  env = control.wrappers.ActionRepeat(env, action_repeat)
  env = control.wrappers.NormalizeActions(env)
  env = control.wrappers.MinimumDuration(env, min_length)
  env = control.wrappers.MaximumDuration(env, max_length)
  if obs_is_image:
    env = control.wrappers.ObservationDict(env, 'image')
    env = control.wrappers.ObservationToRender(env)
  else:
    env = control.wrappers.ObservationDict(env, 'state')
  env = control.wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
  env = control.wrappers.ConvertTo32Bit(env)
  return env

####OSIM

import gym
from gym.spaces.box import Box
from osim.env import L2M2019Env
import functools



def gym_opensim(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward']
  env_ctor = functools.partial(
      _gym_osim_env, action_repeat, config.batch_shape[1], max_length,
      'NIPS2019', obs_is_image=False)
  return Task('gym_opensim', env_ctor, max_length, state_components)



def create_opensim_environment(visualize=False, seed=None, difficulty=2):
  print("Creating Environment .............")
  mode = '3D'
  env = L2M2019Env(visualize=visualize, seed=seed, difficulty=difficulty)
  env.change_model(model=mode, difficulty=difficulty, seed=seed)

  env = OpensimPreprocessing(env)
  num_actions = 22
  env.action_space.n = num_actions
  return env



def _gym_osim_env(action_repeat, min_length, max_length, name, obs_is_image=False):
  env = create_opensim_environment(False)

  env = control.wrappers.ActionRepeat(env, action_repeat)
  env = control.wrappers.NormalizeActions(env)
  env = control.wrappers.MinimumDuration(env, min_length)
  env = control.wrappers.MaximumDuration(env, max_length)
  #if obs_is_image:
  #  env = control.wrappers.ObservationDict(env, 'image')
  #  env = control.wrappers.ObservationToRender(env)
  #else:
  env = control.wrappers.ObservationDict(env, 'state')
  #env = control.wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
  env = control.wrappers.ConvertTo32Bit(env)
  return env



class OpensimPreprocessing(object):
  """A class implementing opensim preprocessing for Prosthetic agents.

  Specifically, this provides the following subset from the JAIR paper
  (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

    * Frame skipping (defaults to 4).
    * Terminal signal when a life is lost (off by default).

  More generally, this class follows the preprocessing guidelines set down in
  Machado et al. (2018), "Revisiting the Arcade Learning Environment:
  Evaluation Protocols and Open Problems for General Agents".
  """

  def __init__(self, environment, frame_skip=4, obs_as_dict=False):
    """Constructor for an Atari 2600 preprocessor.

    Args:
      environment: Gym environment whose observations are preprocessed.
      frame_skip: int, the frequency at which the agent experiences the game.
      terminal_on_life_loss: bool, If True, the step() method returns
        is_terminal=True whenever a life is lost. See Mnih et al. 2015.
      screen_size: int, size of a resized Atari 2600 frame.

    Raises:
      ValueError: if frame_skip or screen_size are not strictly positive.
    """
    if frame_skip <= 0:
      raise ValueError('Frame skip should be strictly positive, got {}'.
                       format(frame_skip))

    self.environment = environment
    self.frame_skip = frame_skip
    self.obs_as_dict = obs_as_dict

    #obs_dims = self.environment.observation_space


    #print("BIKS: obs_dims", obs_dims)

    # Stores temporary observations used for pooling over two successive
    # frames.
    #self.obs_buffer = [
    #    np.empty((obs_dims.shape[0]), dtype=np.float32),
    #    np.empty((obs_dims.shape[0]), dtype=np.float32)
    #]

    self.game_over = False

  @property
  def observation_space(self):
    # Return the observation space adjusted to match the shape of the processed
    # observations.
    return Box(low=0.0, high=1.0, shape=(339,), dtype=np.float32)

  @property
  def action_space(self):
    return self.environment.action_space

  @property
  def reward_range(self):
    return self.environment.reward_range

  @property
  def metadata(self):
    return self.environment.metadata

  def close(self):
    return self.environment.close()

  def reset(self):
    """Resets the environment.

    Returns:
      observation: numpy array, the initial observation emitted by the
        environment.
    """
    INIT_POSE = np.array([
    1.699999999999999956e+00, # forward speed
    .5, # rightward speed
    9.023245653983965608e-01, # pelvis height
    2.012303881285582852e-01, # trunk lean
    0*np.pi/180, # [right] hip adduct
    -6.952390849304798115e-01, # hip flex
    -3.231075259785813891e-01, # knee extend
    1.709011708233401095e-01, # ankle flex
    0*np.pi/180, # [left] hip adduct
    -5.282323914341899296e-02, # hip flex
    -8.041966456860847323e-01, # knee extend
    -1.745329251994329478e-01]) # ankle flex

    sim_dt = 0.01
    sim_t = 10
    timstep_limit = int(round(sim_t/sim_dt))


    obs_dict = self.environment.reset(project=True, seed=None, obs_as_dict=self.obs_as_dict, init_pose=INIT_POSE)
    self.environment.spec.timestep_limit = timstep_limit
    #return np.array(self._obs_dict_to_arr(obs_dict))
    if self.obs_as_dict == False:
      return np.array(obs_dict)
    else:
        return obs_dict  

  def _obs_dict_to_arr(self, obs_dict):
          # Augmented environment from the L2R challenge
    res = []

    # target velocity field (in body frame)
    v_tgt = np.ndarray.flatten(obs_dict['v_tgt_field'])
    res += v_tgt.tolist()

    res.append(obs_dict['pelvis']['height'])
    res.append(obs_dict['pelvis']['pitch'])
    res.append(obs_dict['pelvis']['roll'])
    res.append(obs_dict['pelvis']['vel'][0]/self.environment.LENGTH0)
    res.append(obs_dict['pelvis']['vel'][1]/self.environment.LENGTH0)
    res.append(obs_dict['pelvis']['vel'][2]/self.environment.LENGTH0)
    res.append(obs_dict['pelvis']['vel'][3])
    res.append(obs_dict['pelvis']['vel'][4])
    res.append(obs_dict['pelvis']['vel'][5])

    for leg in ['r_leg', 'l_leg']:
            res += obs_dict[leg]['ground_reaction_forces']
            res.append(obs_dict[leg]['joint']['hip_abd'])
            res.append(obs_dict[leg]['joint']['hip'])
            res.append(obs_dict[leg]['joint']['knee'])
            res.append(obs_dict[leg]['joint']['ankle'])
            res.append(obs_dict[leg]['d_joint']['hip_abd'])
            res.append(obs_dict[leg]['d_joint']['hip'])
            res.append(obs_dict[leg]['d_joint']['knee'])
            res.append(obs_dict[leg]['d_joint']['ankle'])
            for MUS in ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']:
                res.append(obs_dict[leg][MUS]['f'])
                res.append(obs_dict[leg][MUS]['l'])
                res.append(obs_dict[leg][MUS]['v'])
    return res

  def render(self, mode):
    return self.environment.render(mode)

  def step(self, action):
    accumulated_reward = 0.0

    obs_dict, reward, done, info = self.environment.step(action, project = True, obs_as_dict=self.obs_as_dict)
    accumulated_reward += reward

    if self.obs_as_dict == False:
      obs_dict = np.array(obs_dict)
    #observation = np.array(self._obs_dict_to_arr(obs_dict))

    self.game_over = done
    return obs_dict, accumulated_reward, done, info
