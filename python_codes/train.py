import numpy as np
import gymnasium as gym
import torch
import torch.optim as optim
from tqdm import tqdm
from network import *
from utils import *
from Env_wrapper import Atari_Wrapper

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
torch.cuda.empty_cache()


def train_dqn(env_name, network_name, total_steps):

    # hyperparameter
    num_stacked_frames = 4

    replay_memory_size = 250 #250 #26
    min_replay_size_to_update = 25 #25 # this should solve the cuda memory problem (n.2)

    if env_name == 'Spaceinvaders-v5':
        lr = 6e-5 # SpaceInvaders
    else:
        lr = 2.5e-5 # Breakout
    gamma = 0.99
    minibatch_size = 32
    steps_rollout = 16
    updating_steps = 4 

    start_eps = 1
    final_eps = 0.1

    final_eps_frame = 10000
    #total_steps = 200000   

    target_net_update = 100

    save_model_steps = 5000

    # init
    raw_env = gym.make(f'ALE/{env_name}')
    env = Atari_Wrapper(raw_env, env_name, num_stacked_frames, use_add_done=True)

    in_channels = num_stacked_frames
    num_actions = env.action_space.n

    eps_interval = start_eps-final_eps

    agent = Agent(in_channels, num_actions, start_eps).to(device)

    if network_name == 'DDQN':
        target_agent = Agent(in_channels, num_actions, start_eps).to(device)
        target_agent.load_state_dict(agent.state_dict())

    replay = Experience_Replay(replay_memory_size)
    runner = Env_Runner(env, agent)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    huber_loss = torch.nn.SmoothL1Loss()

    num_steps = 0
    num_model_updates = 0

    loss_values = []
    cum_rew = []

    pbar = tqdm(desc='Training', total=total_steps)

    while num_steps < total_steps:
        
        # set agent exploration | cap exploration after x timesteps to final epsilon
        new_epsilon = np.maximum(final_eps, start_eps - ( eps_interval * num_steps/final_eps_frame))
        agent.set_epsilon(new_epsilon)
        
        # get data
        obs, actions, rewards, dones = runner.run(steps_rollout)
        transitions = make_transitions(obs, actions, rewards, dones)
        replay.insert(transitions)
        
        # add
        num_steps += steps_rollout
        pbar.update(steps_rollout)
        
        # check if update
        if num_steps < min_replay_size_to_update:
            continue
        
        # update
        for update in range(updating_steps):
            optimizer.zero_grad()
            
            minibatch = replay.get(minibatch_size)
            
            # uint8 to float32 and normalize to 0-1
            obs = (torch.stack([i[0] for i in minibatch]).to(device).to(dtype)) / 255 
            
            actions = np.stack([i[1] for i in minibatch])
            rewards = torch.tensor([i[2] for i in minibatch]).to(device)
            
            # uint8 to float32 and normalize to 0-1
            next_obs = (torch.stack([i[3] for i in minibatch]).to(device).to(dtype)) / 255
            
            dones = torch.tensor([i[4] for i in minibatch]).to(device)
            
            # prediction
            Qs = agent(torch.cat([obs, next_obs]))
            obs_Q, next_obs_Q = torch.split(Qs, minibatch_size ,dim=0)
            
            obs_Q = obs_Q[range(minibatch_size), actions]
            
            # target
            
            next_obs_Q_max = torch.max(next_obs_Q,1)[1].detach()

            if network_name == 'DDQN':
                target_Q = target_agent(next_obs)[range(minibatch_size), next_obs_Q_max].detach()
                
            else:
                target_Q = next_obs_Q_max

            target = rewards + gamma * target_Q * (1-dones)

            # loss
            loss = huber_loss(obs_Q, target) 
            
            loss_values.append(loss.item())
            epoch_rew = sum(rewards)
            cum_rew.append(epoch_rew)

            loss.backward()
            optimizer.step()
            
        num_model_updates += 1
        
        if network_name == 'DDQN':
            # update target network
            if num_model_updates%target_net_update == 0:
                target_agent.load_state_dict(agent.state_dict())
        
        # save the dqn after some time
        if num_steps%save_model_steps < steps_rollout:
            torch.save(agent,f"{network_name}_checkpoints/{network_name}-{num_steps}.pt")

    pbar.close()
    env.close()

    ## save training information
    saving_folder = 'train_info'
    create_dir(saving_folder)
    with open(f'{saving_folder}/training_loss_{network_name}_{num_steps}.txt', 'w') as f:
        for el in loss_values:
            f.write(f'{el}\n')

    with open(f'{saving_folder}/cumulative_reward_{network_name}_{num_steps}.txt', 'w') as f:
        for el in cum_rew:
            f.write(f'{el}\n')