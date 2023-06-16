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
            self.memory[self.position] = transitions[i] #insert one tuple each time
            self.position = (self.position + 1) % self.capacity

    def get(self, batch_size):
        indexes = (np.random.rand(batch_size) * (len(self.memory)-1)).astype(int) #draw a *batch_size* random tuple from the memory buffer
        return [self.memory[i] for i in indexes] 

    def __len__(self):
        return len(self.memory)

class Env_Runner:
    
    def __init__(self, env, agent):
        super().__init__()
        
        self.env = env
        self.agent = agent
        
        self.ob = self.env.reset() #ob is a stack of size k
        self.total_steps = 0
        
    def run(self, steps, batch_size):
        
        obs = []
        actions = []
        rewards = []
        dones = []

        for step in range(steps):
            
            #self.ob = torch.tensor(self.ob) # uint8
            action = self.agent.e_greedy(self.ob, batch_size) # float32+norm
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
    # observations are in uint8 format
    tuples = []

    steps = len(obs) - 1
    for t in range(steps):
        tuples.append((obs[t],
                       actions[t],
                       rewards[t],
                       obs[t+1], #next_obs
                       int(not dones[t])))
        
    return tuples

def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)