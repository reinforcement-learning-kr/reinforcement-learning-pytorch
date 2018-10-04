import os
import sys
import gym
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from model import Model
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="CartPole-v1", help='')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/', help='')
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--gamma', default=0.99, help='')
parser.add_argument('--log_interval', default=10, help='')
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()


def train_model(sample, policy, value):
    state = sample[0]
    action = sample[1]
    reward = sample[2]
    next_state = sample[3]
    mask = sample[4]

    _, next_value = net(next_state)
    pred = reward + mask * args.gamma * next_value[0]
    td_error = pred - value[0]

    log_policy = torch.log(policy[0])[action]
    loss1 = - log_policy * td_error.item()
    loss2 = F.mse_loss(value[0], pred.detach())
    entropy = torch.log(policy[0]) * policy[0]
    loss = loss1 + loss2 - 0.1 * entropy.sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def get_action(policy):
    policy = policy.data.numpy()[0]
    action = np.random.choice(num_actions, 1, p=policy)[0]
    return action


if __name__=="__main__":
    env = gym.make(args.env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    net = Model(num_inputs, num_actions)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    writer = SummaryWriter('logs')

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    net.train()
    running_score = 0

    for e in range(3000):
        done = False
        score = 0

        state = env.reset()
        state = torch.Tensor(state)
        state = state.unsqueeze(0)

        while not done:
            if args.render:
                env.render()

            policy, value = net(state)
            action = get_action(policy)

            next_state, reward, done, _ = env.step(action)
            next_state = torch.Tensor(next_state)
            next_state = next_state.unsqueeze(0)
            
            mask = 0 if done else 1
            reward = reward if not done or score == 499 else -1
            sample = [state, action, reward, next_state, mask]
            train_model(sample, policy, value)

            score += reward
            state = next_state

        score = score if score == 500.0 else score + 1
        running_score = 0.99 * running_score + 0.01 * score
        if e % args.log_interval == 0:
            print('{} episode score: {:.2f}'.format(e, running_score))
            writer.add_scalar('log/score', float(score), running_score)

        if running_score > env.spec.reward_threshold:
            running_score = int(running_score)
            ckpt_path = model_path + 'ckpt_'+ str(e)+'.pth'
            torch.save(net.state_dict(), ckpt_path)
            print('running score exceeds 485 so end')
            break



    

