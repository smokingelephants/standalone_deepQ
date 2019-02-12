import gym

#from baselines import deepq
import models  # noqa
from build_graph import build_act, build_train  # noqa
from simple import learn, load  # noqa
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer  # noqa
import pickle

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():

    env = gym.make("CartPole-v0")
    #env = gym.make("MountainCar-v0")
    model = models.mlp([256, 20])
    act = learn(
        env,
        q_func=model,
        lr=1e-2,
        max_timesteps=100000,
        buffer_size=90000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        print_freq=25,
        checkpoint_path='model_chkpoints/cart_model',
        callback=callback,
        param_noise=True
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
