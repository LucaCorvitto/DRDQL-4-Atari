import os
import torch
import numpy as np
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

class Experience_Replay():

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def insert(self, transitions):
        
        for i in range(len(transitions)):
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = transitions[i]
            self.position = (self.position + 1) % self.capacity

    def get(self, batch_size):
        #return random.sample(self.memory, batch_size)
        indexes = (np.random.rand(batch_size) * (len(self.memory)-1)).astype(int)
        return [self.memory[i] for i in indexes]

    def __len__(self):
        return len(self.memory)

class Env_Runner:
    
    def __init__(self, env, agent):
        super().__init__()
        
        self.env = env
        self.agent = agent
        
        self.ob = self.env.reset()
        self.total_steps = 0
        
    def run(self, steps):
        
        obs = []
        actions = []
        rewards = []
        dones = []
        
        for step in range(steps):
            
            self.ob = torch.tensor(self.ob) # uint8
            action = self.agent.e_greedy(
                self.ob.to(device).to(dtype).unsqueeze(0) / 255) # float32+norm
            action = action.detach().cpu().numpy()
            
            obs.append(self.ob)
            actions.append(action)
            
            self.ob, r, done, additional_done, info = self.env.step(action)
               
            if done: # real environment reset, other add_dones are for q learning purposes
                self.ob = self.env.reset()
                if "return" in info:
                    self.logger.log(f'{self.total_steps+step},{info["return"]}')
            
            rewards.append(r)
            dones.append(done or additional_done)
            
        self.total_steps += steps
                                    
        return obs, actions, rewards, dones
    
def make_transitions(obs, actions, rewards, dones):
    
    tuples = []

    steps = len(obs) - 1
    for t in range(steps):
        tuples.append((obs[t],
                       actions[t],
                       rewards[t],
                       obs[t+1],
                       int(not dones[t])))
        
    return tuples

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)