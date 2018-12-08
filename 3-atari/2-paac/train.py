import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(net, optimizer, transition, args):
    histories = torch.stack(transition.history).to(device)
    next_histories = torch.stack(transition.next_history).to(device)
    actions = torch.Tensor(transition.action).long().to(device)
    rewards = torch.Tensor(transition.reward).to(device)
    masks = torch.Tensor(transition.mask).to(device)

    policies, values = net(histories[0])
    _, last_values = net(next_histories[-1])

    # get multi-step td-error
    running_returns = last_values.squeeze(-1)
    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + args.gamma * running_returns * masks[t]

    preds = running_returns
    td_errors = (preds - values.squeeze(-1)).detach()
    
    # get policy gradient
    log_policies = torch.log(policies.gather(1, actions[0].view(-1,1)) + 1e-5)

    loss_p = - log_policies * td_errors
    loss_v= F.mse_loss(values.squeeze(-1), preds.detach())
    entropy = - policies * torch.log(policies + 1e-5)
    loss = loss_p.mean() + args.value_coef * loss_v.mean() - args.entropy_coef * entropy.detach().mean()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)
    optimizer.step()
