import os
import gym
import random
import argparse
import numpy as np

import torch
from utils import pre_process, get_action
from model import QNet

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="BreakoutDeterministic-v4", help='')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/', help='')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    env = gym.make(args.env_name)
    env.seed(500)
    torch.manual_seed(500)

    img_shape = env.observation_space.shape
    num_actions = 3
    print('image size:', img_shape)
    print('action size:', num_actions)

    net = QNet(num_actions)
    net.load_state_dict(torch.load(args.save_path + 'model.pth'))
    
    net.to(device)
    net.eval()

    epsilon = 0

    for e in range(5):
        done = False

        score = 0
        state = env.reset()

        state = pre_process(state)
        state = torch.Tensor(state).to(device)
        history = torch.stack((state, state, state, state))

        for i in range(3):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            state = pre_process(state)
            state = torch.Tensor(state).to(device)
            state = state.unsqueeze(0)
            history = torch.cat((state, history[:-1]), dim=0)

        while not done:
            if args.render:
                env.render()

            steps += 1
            qvalue = net(history.unsqueeze(0))
            action = get_action(0, qvalue, num_actions)

            next_state, reward, done, info = env.step(action + 1)
            
            next_state = pre_process(next_state)
            next_state = torch.Tensor(next_state).to(device)
            next_state = next_state.unsqueeze(0)
            next_history = torch.cat((next_state, history[:-1]), dim=0)
            
            score += reward
            history = next_history

        print('{} episode | score: {:.2f}'.format(e, score))


if __name__=="__main__":
    main()
   