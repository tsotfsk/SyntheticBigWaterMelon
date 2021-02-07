from collections import namedtuple
import json
from easydict import EasyDict as edict

from game.env import Game
from model.dqn import DQN
import torch

# Hyper-parameters
seed = 1
num_episodes = 2000
with open("./game/style.json", 'r') as f:
    args = edict(json.load(f))
env = Game(args)
torch.manual_seed(seed)

Transition = namedtuple(
    'Transition', ['state', 'ball', 'action', 'reward', 'next_state', 'next_ball'])
"""
The main loop of the game.
:return: None
"""


def main():

    agent = DQN()
    for i_ep in range(num_episodes):
        env = Game(args)
        env.seed(seed)
        state, ball = env.start_obs()
        state = torch.from_numpy(state).float()
        while env._running:
            action = agent.select_action(state, ball)
            next_state, next_ball, reward, done = env.step(env.action_space[action])
            next_state = torch.from_numpy(next_state).float()
            transition = Transition(state, ball, action, reward, next_state, next_ball)
            agent.store_transition(transition)
            state = next_state
            ball = next_ball
            env._clock.tick(env._fps)
            if done:
                loss = agent.update()
                print("episodes {}, total reward is {}".format(i_ep, env._score) + 
                    " total frame is {}, total loss is {} ".format(env._frame, loss if loss is not None else "None"))
                break


if __name__ == '__main__':
    main()
