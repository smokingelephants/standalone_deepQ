import models  # noqa
from build_graph import build_act, build_train  # noqa
from simple import learn, load  # noqa
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa

#def wrap_atari_dqn(env):
#    from baselines.common.atari_wrappers import wrap_deepmind
#    return wrap_deepmind(env, frame_stack=True, scale=True)