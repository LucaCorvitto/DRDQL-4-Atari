import gymnasium as gym
import torch
from torchvision import transforms
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

class Atari_Wrapper(gym.Wrapper):
    # env wrapper to resize images, grey scale and frame stacking and other misc.
    
    def __init__(self, env, env_name, k, L, dsize=(84,84), use_add_done=False):
        super(Atari_Wrapper, self).__init__(env)
        self.dsize = dsize
        self.k = k
        self.L = L
        self.use_add_done = use_add_done
        self.gs = transforms.Grayscale()
        self.rs = transforms.Resize(dsize)
        
        # set image cutout depending on game
        if "SpaceInvaders" in env_name: #if 'bellaaaaa' in env_name:
            self.frame_cutout_h = (25,-7)
            self.frame_cutout_w = (7,-7)
        else:
            # no cutout
            self.frame_cutout_h = (0,-1)
            self.frame_cutout_w = (0,-1)
        
    def reset(self):
    
        self.Return = 0
        self.last_life_count = 0
        
        ob, _ = self.env.reset() #I think here it calls the default version of the reset function
        ob = self.preprocess_observation(ob) #now ob is ready to be feed in the network
        #print('observation: ', ob.shape)
        
        # create L stacks of the reset observation
        self.frame_stack_list = [] #reset the frame list; it will be of size self.L
        #for i in range(self.L): #iterate over the seq_len
            #frame_stack = np.stack([ob for j in range(self.k)]) #to create L couples (k times) of stacks
        frame_stack = torch.stack([ob for j in range(self.k)])
        #self.frame_stack_list.append(frame_stack)
        #print(frame_stack.shape)
        #To parallelize the process create a single stack of the entire sequence
        #self.frame_stack_sequence = np.stack(self.frame_stack_list)
        self.frame_stack_sequence = torch.stack([frame_stack for i in range(self.L)])

        #print(self.frame_stack_sequence.shape)

        return self.frame_stack_sequence #instead of a list of stacks returns a stack of the entire sequence
    
    
    def step(self, action): 
        # do k frameskips, same action for every intermediate frame
        # stacking k frames
        
        reward = 0
        done = False
        additional_done = False
        
        # k frame skips or end of episode
        frames = []
        stacks = []
        tot_frames = self.L * 2
        for i in range(tot_frames):
            # build observation during the step in order to reduce the computational cost
            
            ob, r, d, _, info = self.env.step(action) #I think here it calls the default version of the step function #1 step
            
            # insert a (additional) done, when agent loses a life (Games with lives)
            if self.use_add_done:
                if info['lives'] < self.last_life_count:
                    additional_done = True  
                self.last_life_count = info['lives']
            
            #print('ob info: ', type(ob), ob)
            processed_ob = self.preprocess_observation(ob) #prepare frames to be feed into the network (into device)
            frames.append(processed_ob)

            if (i+1)%self.k==0: #collected enough frames
                frame_stack = torch.stack(frames)
                stacks.append(frame_stack)
                frames = [] #reset the frames list
            
            # add reward
            reward += r
            
            if d: # env done
                done = True
                break
                       
        # build the observation
        num_frames = len(frames) #max is k, but then is reset, so check for 0
        num_stacks = len(stacks) #max is L
        # The episode can end prematurely and we have to consider these cases. There are 3:
        # 1st: len(frames)==0, but i<tot_frames; in this case we have a slot of n stacks that we can add to an existing frame_stack_sequence
        if num_frames==0 and (i+1)!=tot_frames:
            front = self.frame_stack_sequence[num_stacks::]   # /oppure/ self.frame_stack_sequence[0: self.L - num_stacks] = self.frame_stack_sequence[num_stacks::]
            curr_stack_seq = torch.stack(stacks)
            self.frame_stack_sequence = torch.cat([front, curr_stack_seq])    # /oppure/ self.frame_stack[self.L - num_stacks::] =  curr_stack_seq
            #print('frame_stack_seq dim: ', self.frame_stack_sequence.shape)
        # 2nd: len(frames)>0 for the last stack; in this case we have to mask the remaing elements to fill the last stack
        elif num_frames>0 and (i+1)==tot_frames:
            mask_tensor = torch.zeros_like(processed_ob)
            #create the last stack
            for i in range(self.k-num_frames):
                frames.append(mask_tensor)
            last_stack = torch.stack(frames)
            # adding last stack to the list of stacks and create the sequence
            stacks.append(last_stack)
            self.frame_stack_sequence = torch.stack(stacks)
        # 3rd: len(frames)>0 and i<tot_frames; in this case we have to do both
        elif num_frames>0 and (i+1)!=tot_frames: #episode ended before collecting all the frames, this happen when the stack exists yet
            # move the last frames of the old sequence to the front=0
            front = self.frame_stack_sequence[num_stacks::]
            #create the last stack
            mask_tensor = torch.zeros_like(processed_ob)
            for i in range(self.k-num_frames):
                frames.append(mask_tensor)
            last_stack = torch.stack(frames)
            # adding last stack to the list of stacks and create the sequence
            stacks.append(last_stack)
            curr_stack_seq = torch.stack(stacks)
            self.frame_stack_sequence = torch.cat([front, curr_stack_seq])
        else: #episode end without any problem, so we just create a new stack sequence
            self.frame_stack_sequence = torch.stack(stacks)
        
        # add info, get return of the completed episode
        self.Return += reward
        if done:
            info["return"] = self.Return
            
        # clip reward
        if reward > 0:
            reward = 1 #maybe I could leave this reward as it is
        elif reward == 0:
            reward = 0
        else:
            reward = -1
            
        return self.frame_stack_sequence, reward, done, additional_done, info

    def preprocess_observation(self, ob):
    # resize and grey and cutout image

        # to tensor
        ob = ob[self.frame_cutout_h[0]:self.frame_cutout_h[1], self.frame_cutout_w[0]:self.frame_cutout_w[1]].transpose(2,0,1) #Torch wants images in format (channels, height, width) 
        ob = torch.from_numpy(ob) #maybe torch.tensor(ob) is enough
        #print('tensor ob: ', ob.shape)
        #reduce channel from 3 to 1
        ob = self.gs(ob)
        #reduce image size to the specified one (84x84)
        ob = self.rs(ob)
        #transfer to gpu and change type
        ob = ob.view(-1,ob.size(2)).to(device).to(dtype)
        #normalize
        ob = ob/255


        # #reduce channel from 3 to 1
        # ob = cv2.cvtColor(ob[self.frame_cutout_h[0]:self.frame_cutout_h[1],
        #                    self.frame_cutout_w[0]:self.frame_cutout_w[1]], cv2.COLOR_BGR2GRAY)
        # #reduce image size to the specified one (80x80)
        # ob = cv2.resize(ob, dsize=self.dsize)
    
        return ob