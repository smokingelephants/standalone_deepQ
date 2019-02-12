import gym
import os
from baselines import deepq
import numpy as np
import pickle
import random
def main():
    env = gym.make("CartPole-v0")
    act = deepq.load("cartpole_model.pkl")

    episode=0
    chain_dump=[]
    trans=[]
    while episode<=10:    # True:
        obs, done = env.reset(), False
        episode_rew = 0
        trans=[]
        while not done:
            env.render()
            r=random.uniform(0,1)
            if(r <= 0.55):
                action = act(obs[None])[0]
            else:
                action = random.randint(0,1)
            new_obs, rew, done, _ = env.step(action)

            trans.append([obs, action, rew, new_obs])
            obs=new_obs
            episode_rew += rew
        episode=episode+1
        print("Episode reward", episode_rew)
        chain_dump.append(np.vstack(trans))

    filehandler = open("policy_transitions.seq","wb")
    pickle.dump(chain_dump,filehandler, protocol=1)
    filehandler.close()
    print('policy sequences saved',replay_buffer.__len__())
    print('done')


if __name__ == '__main__':
    main()
