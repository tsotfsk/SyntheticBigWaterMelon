from collections import namedtuple

from game import Game, FPS
from model.dqn import DQN
import torch

# Hyper-parameters
seed = 1
env = Game()
num_episodes = 2000
torch.manual_seed(seed)

Transition = namedtuple(
    'Transition', ['state', 'ball', 'action', 'reward', 'next_state', 'next_ball'])


def main():

    agent = DQN()
    for i_ep in range(num_episodes):
        env.reset()
        env.seed(seed)
        state, ball = env.start_obs()
        state = torch.from_numpy(state).float()
        while True:
            action = agent.select_action(state, ball)
            next_state, next_ball, reward, done = env.step(
                env.action_space[action])
            next_state = torch.from_numpy(next_state).float()
            transition = Transition(
                state, ball, action, reward, next_state, next_ball)
            agent.store_transition(transition)
            state = next_state
            ball = next_ball
            env.clock.tick(FPS)
            if done:
                loss = agent.update()
                print("episodes {}, total reward is {}".format(i_ep, env.score) +
                      " total frame is {}, total loss is {} ".format(env.frame, loss if loss is not None else "None"))
                break


if __name__ == '__main__':
    main()
