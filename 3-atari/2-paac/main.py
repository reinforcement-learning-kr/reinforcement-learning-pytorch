import os
import gym
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.multiprocessing import Pipe

from model import ActorCritic
from utils import get_action
from train import train_model
from env import EnvWorker
from memory import Memory

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="MsPacman-v4", help='')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/', help='')
parser.add_argument('--render', default=False, action="store_true")
parser.add_argument('--gamma', default=0.99, help='')
parser.add_argument('--goal_score', default=400, help='')
parser.add_argument('--log_interval', default=10, help='')
parser.add_argument('--save_interval', default=1000, help='')
parser.add_argument('--num_envs', default=8, help='')
parser.add_argument('--num_step', default=5, help='')
parser.add_argument('--value_coef', default=0.5, help='')
parser.add_argument('--entropy_coef', default=0.01, help='')
parser.add_argument('--lr', default=7e-4, help='')
parser.add_argument('--eps', default=1e-5, help='')
parser.add_argument('--clip_grad_norm', default=0.5, help='')
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    env = gym.make(args.env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    net = ActorCritic(num_actions)
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, eps=args.eps)
    writer = SummaryWriter('logs')

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    workers = []
    parent_conns = []
    child_conns = []
    
    # make running environments for workers and 
    # pipelines for connection between agent and environment
    for i in range(args.num_envs):
        parent_conn, child_conn = Pipe()
        worker = EnvWorker(args.env_name, args.render, child_conn)
        worker.start()
        workers.append(worker)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)
    
    net.to(device)
    net.train()
    
    global_steps = 0
    score = 0
    count = 0
    
    histories = torch.zeros([args.num_envs, 4, 84, 84]).to(device)

    while True:
        count += 1
        memory = Memory(capacity=args.num_step)
        global_steps += (args.num_envs * args.num_step)
    
        # gather samples from environment
        for i in range(args.num_step):
            policies, values = net(histories.to(device))
            actions = get_action(policies, num_actions)

            # send action to each worker environement and get state information
            next_histories, rewards, masks, dones = [], [], [], []
            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)
                next_history, reward, dead, done = parent_conn.recv()
                next_histories.append(next_history.unsqueeze(0))
                rewards.append(reward)
                masks.append(1-dead)
                dones.append(done)

            score += rewards[0]

            # if agent in first environment dies, print and log score
            if dones[0]:
                print('global steps {} | score: {}'.format(global_steps, score))
                writer.add_scalar('log/score', score, global_steps)
                score = 0

            next_histories = torch.cat(next_histories, dim=0)
            rewards = np.hstack(rewards)
            masks = np.hstack(masks)

            memory.push(histories.cpu(), next_histories.cpu(), actions, rewards, masks)
            histories = next_histories

        # train network with accumulated samples
        transitions = memory.sample()
        train_model(net, optimizer, transitions, args)

        if count % args.save_interval == 0:
            ckpt_path = args.save_path + 'model.pt'
            torch.save(net.state_dict(), ckpt_path)


if __name__=="__main__":
    main()
    