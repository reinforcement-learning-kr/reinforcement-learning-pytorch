import os
import sys
import gym
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from memory import Memory
from model import ActorCritic
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="CartPole-v1", help='')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/', help='')
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--gamma', default=0.99, help='')
parser.add_argument('--goal_score', default=400, help='')
parser.add_argument('--log_interval', default=10, help='')
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(net, optimizer, batch):
    states = torch.stack(batch.state).to(device)
    next_states = torch.stack(batch.next_state).to(device)
    actions = torch.Tensor(batch.action).long().to(device)
    rewards = torch.Tensor(batch.reward).to(device)
    masks = torch.Tensor(batch.mask).to(device)

    policy, value = net(states[0])
    _, last_value = net(next_states[-1])

    running_returns = last_value[0]
    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + args.gamma * running_returns * masks[t]

    pred = running_returns
    td_error = pred - value[0]

    log_policy = torch.log(policy[0] + 1e-5)[actions[0]]
    loss1 = - log_policy * td_error.item()
    loss2 = F.mse_loss(value[0], pred.detach())
    entropy = torch.log(policy + 1e-5) * policy
    loss = loss1 + loss2 - 0.01 * entropy.sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def get_action(policy, num_actions):
    policy = policy.data.numpy()[0]
    action = np.random.choice(num_actions, 1, p=policy)[0]
    return action


def main():
    env = gym.make(args.env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    net = ActorCritic(num_inputs, num_actions)
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    writer = SummaryWriter('logs')

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    net.to(device)
    net.train()
    memory = Memory(100)
    running_score = 0

    for e in range(3000):
        done = False
        score = 0
        steps = 0

        state = env.reset()
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)

        while not done:
            if args.render:
                env.render()

            steps += 1
            policy, value = net(state)
            action = get_action(policy, num_actions)

            next_state, reward, done, _ = env.step(action)
            next_state = torch.Tensor(next_state).to(device)
            next_state = next_state.unsqueeze(0)
            
            mask = 0 if done else 1
            reward = reward if not done or score == 499 else -1
            
            memory.push(state, next_state, action, reward, mask)

            if len(memory) == 5 or done:
                batch = memory.sample()
                train_model(net, optimizer, batch)
                memory = Memory(100)

            score += reward
            state = next_state
        
        score = score if score == 500.0 else score + 1
        running_score = 0.99 * running_score + 0.01 * score
        if e % args.log_interval == 0:
            print('{} episode | score: {:.2f}'.format(e, running_score))
            writer.add_scalar('log/score', float(score), running_score)

        if running_score > args.goal_score:
            ckpt_path = args.save_path + 'model.pth'
            torch.save(net.state_dict(), ckpt_path)
            print('running score exceeds 400 so end')
            break


if __name__=="__main__":
    main()
