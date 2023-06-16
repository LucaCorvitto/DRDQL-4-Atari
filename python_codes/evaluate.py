import gymnasium as gym
import torch
from tqdm import tqdm
import time
import numpy as np
from network import *
from utils import create_dir
from Env_wrapper import Atari_Wrapper

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
torch.cuda.empty_cache()

def evaluate(env_name, network_name, trained, rendering, total_steps):

    #training hyperparameters
    num_stacked_frames = 4

    # watch
    agent_name = f"{network_name}-{total_steps}.pt"
    if trained:
        torch.save(agent,agent_name)
        agent = torch.load(agent_name)
    else:
        agent = torch.load(f"{network_name}_checkpoints/{agent_name}")
    
    print(f'Evaluating agent {agent_name}...')

    render = 'human' if rendering else 'rgb_array'
    raw_env = gym.make(f'ALE/{env_name}', render_mode=render)
    env = Atari_Wrapper(raw_env, env_name, num_stacked_frames)

    steps = 5000
    ob = env.reset()
    agent.set_epsilon(0.025)
    agent.eval()
    imgs = []
    return_values = []
    
    for step in tqdm(range(steps)):
        action = agent.e_greedy(torch.tensor(ob, dtype=dtype).unsqueeze(0).to(device) / 255)
        action = action.detach().cpu().numpy()

        env.render()
        ob, reward, done, _, info = env.step(action)

        if rendering:
            time.sleep(0.016)

        if done:
            ob = env.reset()
            print(info)
            return_values.append(info['return'])
        
        imgs.append(ob)
        
    env.close()

    saving_folder = 'eval_returns'
    create_dir(saving_folder)
    with open(f'{saving_folder}/{agent_name}.txt', 'w') as f:
        for el in return_values:
            f.write(f'{el}\n')

    mean_return_value = np.mean(return_values)
    print(mean_return_value)
    with open(f'{saving_folder}/{agent_name}_mean_return.txt', 'w') as f:
        f.write(str(mean_return_value))