import numpy as np
import gymnasium as gym
import torch
import torch.optim as optim
from tqdm import tqdm

from network_drqn import *
from utils_rnn import *
from Env_wrapper_rnn import Atari_Wrapper

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
torch.cuda.empty_cache()



def train(env_name, network_name, total_steps):

    # hyperparameter
    num_stacked_frames = 2
    seq_len = 16 

    replay_memory_size = 250
    min_replay_size_to_update = 25

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

    save_model_steps = 5000

    torch.cuda.empty_cache()

    # init
    raw_env = gym.make(f'ALE/{env_name}') 
    env = Atari_Wrapper(raw_env, env_name, num_stacked_frames, seq_len, use_add_done=True)

    in_channels = num_stacked_frames
    num_actions = env.action_space.n

    eps_interval = start_eps-final_eps

    agent = Agent(in_channels, num_actions, start_eps, seq_len, minibatch_size).to(device)

    replay = Experience_Replay(replay_memory_size)
    runner = Env_Runner(env, agent)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    huber_loss = torch.nn.SmoothL1Loss() #more exactly a variation of the huber loss

    num_steps = 0
    num_model_updates = 0

    loss_values = []
    cum_rew = []

    pbar = tqdm(desc='Training', total=total_steps)

    while num_steps < total_steps:
        
        # set agent exploration | cap exploration after x timesteps to final epsilon
        new_epsilon = np.maximum(final_eps, start_eps - ( eps_interval * num_steps/final_eps_frame))
        agent.set_epsilon(new_epsilon)
        agent.reset_hidden_state()
        
        ## get data
        # obs is a list of sequence stacks each of shape [seq_len, n_frames, w_img, h_img]
        obs, actions, rewards, dones = runner.run(steps_rollout, minibatch_size)
        # returns a tuple object containing the informations of each step, and each step is made by k frames
        transitions = make_transitions(obs, actions, rewards, dones)
        # keep memory of all the transitions
        replay.insert(transitions)
        
        ## add
        num_steps += steps_rollout
        pbar.update(steps_rollout)

        ## check if update
        if num_steps < min_replay_size_to_update:
            continue

        ## update
        for update in range(updating_steps):
            optimizer.zero_grad()
        
            # get a minibatch of the transitions
            minibatch = replay.get(minibatch_size)

            obs = torch.stack([i[0] for i in minibatch])
            next_obs = torch.stack([i[3] for i in minibatch])
            
            actions = np.stack([i[1] for i in minibatch])
            rewards = torch.tensor([i[2] for i in minibatch]).to(device)
            dones = torch.tensor([i[4] for i in minibatch]).to(device)
            
            # prediction
            Qs = agent(obs, minibatch_size, next_obs=next_obs)
            obs_Q, next_obs_Q = torch.split(Qs, minibatch_size ,dim=0)

            obs_Q_final = obs_Q[range(minibatch_size), actions]

            next_obs_Q_max = torch.max(next_obs_Q,1)[1].detach()

            # target           
            target = rewards + gamma * next_obs_Q_max * (1-dones)

            # loss
            loss = huber_loss(obs_Q_final, target)

            loss_values.append(loss.item())
            epoch_rew = sum(rewards)
            cum_rew.append(epoch_rew)

            loss.backward()
            optimizer.step()
        
        num_model_updates += 1
        
        # save the dqn after some time
        if num_steps%save_model_steps == 0:
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