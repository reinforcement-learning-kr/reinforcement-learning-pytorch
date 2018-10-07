import torch
import random
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize


def pre_process(image):
    image = np.array(image)
    image = resize(image, (84, 84, 3))
    image = rgb2gray(image)
    return image


def get_action(epsilon, qvalue, num_actions):
    if np.random.rand() <= epsilon:
        return random.randrange(num_actions)
    else:
        _, action = torch.max(qvalue, 1)
        return action.numpy()[0]


def update_target_model(net, target_net):
    target_net.load_state_dict(net.state_dict())


def to_tensor(array):
    tensor = torch.Tensor(array)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor
