import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCNN(nn.Module):

    def __init__(self, n_frames, hidden_size=16):
        super().__init__()
        # changing the shape of the network
        self.conv1 = nn.Conv2d(n_frames, hidden_size, kernel_size=19, stride=8, padding=0)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size*2, kernel_size=8, stride=4, padding=0)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        return x2

class MyLSTM(nn.Module):
    def __init__(self, n_actions, seq_len, hidden_size=16, lstm_size=256, n_layers = 1, bias=True):
        super().__init__()

        self.n_hidden = lstm_size
        self.seq_len = seq_len
        self.n_layers = n_layers
        #the following is related to the output size of the last convolutional layer as it will be the input of the lstm
        self.n_features = hidden_size*2

        self.flatten = nn.Flatten()

        self.lstm = nn.LSTM(input_size=self.n_features, hidden_size=self.n_hidden, num_layers=n_layers, batch_first=True)

        self.fc = nn.Linear(self.n_hidden*self.seq_len, n_actions)


    def forward(self, x, hidden):

        # input x must have shape: [batch_size,seq_len,features]
        # hidden[0] must have shape: [n_layers,batch_size,n_hidden]
        lstm_out, hidden = self.lstm(x, hidden)

        # Flatten the output tensor
        flat_lstm_out = self.flatten(lstm_out)
        # Select last timestep output for Q-value estimation
        last_time_step = flat_lstm_out.view(flat_lstm_out.size(0), -1)


        q_values = self.fc(last_time_step)  # Select last timestep output for Q-value estimation

        return q_values, hidden

class Agent(nn.Module):

    def __init__(self, in_channels, num_actions, epsilon, seq_len, batch_size, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()

        self.device = device
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.seq_len = seq_len
        self.cnn_net = MyCNN(in_channels).to(self.device)
        self.lstm_net = MyLSTM(num_actions, seq_len=self.seq_len).to(self.device)

        self.hidden = self.reset_hidden_state()

        self.eps = epsilon

    def forward(self, obs, batch_size, next_obs=None): #this function has to handle parallelize computing

        ### Parallelized over sequence length and batch_size ###

        # dimension of a stack: [seq_len, n_frames, w_img, h_img]
        if next_obs is not None:
            #  handle batch size and sequence lenght both in the same dimension for both obs
            #unify first 2 dimensions
            curr_x = obs.view(-1, obs.size(2), obs.size(3), obs.size(4))
            #unify first 2 dimensions
            next_x = next_obs.view(-1, next_obs.size(2), next_obs.size(3), next_obs.size(4))
            # concatenate the observations
            x = torch.cat([curr_x, next_x])
        else:
            x = obs

        #these features collects all the information of a sequence
        features = self.cnn_net(x)

        ### Reshape Tensor in order to separate batch_size and sequence_length ###
        
        # Reshape features to (batch_size, timesteps, features)
        pre_lstm = features.view(features.size(0), features.size(1), -1)
        # if seq_len is treated as batch_size I have to permute it in a different order

        if next_obs is not None:
            ext_list = []
            # collect different stacks for each sequence length
            for i in range(batch_size*2):
                extracted = pre_lstm[self.seq_len*i:self.seq_len*i+self.seq_len]
                per_extracted = extracted.permute(2,0,1)
                ext_list.append(per_extracted)

            # create a unique tensor for all the sequences one after the other in the new shape
            lstm_in = torch.cat([el_ext for el_ext in ext_list])
        else:
            to_mask = pre_lstm.permute(2,0,1)
            zeros = torch.zeros_like(to_mask)
            mask = torch.cat([zeros for i in range(batch_size*2-1)])
            lstm_in = torch.cat([to_mask,mask])
            
        qvals, self.hidden = self.lstm_net(lstm_in, self.hidden)
        self.hidden = [_.detach() for _ in self.hidden] #needed to train the lstm: the hidden state must be detached from the backward graph

        return qvals

    def e_greedy(self, obs, batch_size):

        qvals = self.forward(obs, batch_size)

        greedy = torch.rand(1)
        if self.eps < greedy:
            action = qvals.argmax(-1)[0]
            return action
        else:
            return (torch.rand(1) * self.num_actions).type('torch.LongTensor')[0]

    def set_epsilon(self, epsilon):
        self.eps = epsilon

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.lstm_net.n_layers, self.batch_size*2, self.lstm_net.n_hidden).to(self.device), #hidden state h (h,c)
            torch.zeros(self.lstm_net.n_layers, self.batch_size*2, self.lstm_net.n_hidden).to(self.device)  #cell state c   (h,c)
        )
        return self.hidden