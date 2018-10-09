import os
import sys
import gym
import random
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import pre_process, get_action, update_target_model
from model import QNet
from memory import Memory
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="BreakoutDeterministic-v4", help='')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/', help='')
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--gamma', default=0.99, help='')
parser.add_argument('--batch_size', default=32, help='')
parser.add_argument('--initial_exploration', default=1000, help='')
parser.add_argument('--update_target', default=10000, help='')
parser.add_argument('--log_interval', default=1, help='')
parser.add_argument('--goal_score', default=300, help='')
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(net, target_net, optimizer, batch):
    history = torch.stack(batch.history).to(device)
    next_history = torch.stack(batch.next_history).to(device)
    actions = torch.Tensor(batch.action).long().to(device)
    rewards = torch.Tensor(batch.reward).to(device)
    masks = torch.Tensor(batch.mask).to(device)

    pred = net(history).squeeze(1)
    next_pred = target_net(next_history).squeeze(1)
    one_hot_action = torch.zeros(args.batch_size, pred.size(-1))
    one_hot_action = one_hot_action.to(device)
    one_hot_action.scatter_(1, actions.unsqueeze(1), 1)
    pred = torch.sum(pred.mul(one_hot_action), dim=1)
    target = rewards + args.gamma * next_pred.max(1)[0] * masks
    
    loss = F.smooth_l1_loss(pred, target.detach(), size_average=True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.cpu().data


def main():
    env = gym.make(args.env_name)
    env.seed(500)
    torch.manual_seed(500)

    img_shape = env.observation_space.shape
    num_actions = 3
    print('image size:', img_shape)
    print('action size:', num_actions)

    net = QNet(num_actions)
    target_net = QNet(num_actions)
    update_target_model(net, target_net)

    optimizer = optim.RMSprop(net.parameters(), lr=0.00025, eps=0.01)
    writer = SummaryWriter('logs')

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    net.to(device)
    target_net.to(device)
    net.train()
    target_net.train()
    memory = Memory(100000)
    running_score = 0
    epsilon = 1.0
    steps = 0
    
    for e in range(10000):
        done = False
        dead = False

        score = 0
        avg_loss = []
        start_life = 5
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
            action = get_action(epsilon, qvalue, num_actions)

            next_state, reward, done, info = env.step(action + 1)
            
            next_state = pre_process(next_state)
            next_state = torch.Tensor(next_state).to(device)
            next_state = next_state.unsqueeze(0)
            next_history = torch.cat((next_state, history[:-1]), dim=0)

            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']
            
            score += reward
            reward = np.clip(reward, -1, 1)

            mask = 0 if dead else 1
            memory.push(history.cpu(), next_history.cpu(), action, reward, mask)
            
            if dead:
                dead = False
            
            if steps > args.initial_exploration:
                epsilon -= 1e-6
                epsilon = max(epsilon, 0.1)

                batch = memory.sample(args.batch_size)
                loss = train_model(net, target_net, optimizer, batch)

                if steps % args.update_target:
                    update_target_model(net, target_net)
            else:
                loss = 0
                
            avg_loss.append(loss)
            history = next_history


        if e % args.log_interval == 0:
            print('{} episode | score: {:.2f} | epsilon: {:.4f} | steps: {} | loss: {:.4f}'.format(
                e, score, epsilon, steps, np.mean(avg_loss)))
            writer.add_scalar('log/score', float(score), steps)
            writer.add_scalar('log/score', np.mean(avg_loss), steps)

        if score > args.goal_score:
            ckpt_path = args.save_path + 'model.pth'
            torch.save(net.state_dict(), ckpt_path)
            print('running score exceeds 400 so end')
            break

if __name__=="__main__":
    main()