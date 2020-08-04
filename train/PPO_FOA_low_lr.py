import torch
import torch.nn as nn
from torch.distributions import Categorical
from heuristic import Multi_TSP_Heuristic
import matplotlib.pyplot as plt
import gym
import matsp
from utils import *
from wrappers import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        
        k_1, s_1, p_1, c_1 = 4, 2, 1, 8
        self.conv1 = nn.Conv2d(state_dim[2]+2, c_1, k_1, stride = s_1, padding = p_1)
        self.bn1 = nn.BatchNorm2d(c_1)

        k_2, s_2, p_2, c_2 = 4, 2, 1, 16
        self.conv2 = nn.Conv2d(c_1, c_2, k_2, stride = s_2, padding = p_2)
        self.bn2 = nn.BatchNorm2d(c_2)
        w_2, h_2, _ = conv_shape_calc(conv_shape_calc(state_dim, k_1, s_1, p_1),k_2, s_2, p_2) 
        self.conv_latent_layer = w_2*h_2*c_2

        # w_1, h_1, _ = conv_shape_calc(state_dim, k_1, s_1, p_1)
        # self.conv_latent_layer = w_1*h_1*c_1

        # self.conv_latent_layer = state_dim[0]*state_dim[1]*state_dim[2]

        self.debug = nn.Sequential(
                self.conv1,
                self.bn1,
                nn.ReLU(),
                self.conv2,
                self.bn2,
                nn.ReLU()
                )
        # actor
        self.action_layer = nn.Sequential(
                self.conv1,
                self.bn1,
                nn.ReLU(),
                self.conv2,
                self.bn2,
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(self.conv_latent_layer, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.value_layer = nn.Sequential(
                self.conv1,
                self.bn1,
                nn.ReLU(),
                self.conv2,
                self.bn2,
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(self.conv_latent_layer, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.ReLU(),
                nn.Linear(n_latent_var, 1)
                )
        
    # def conv(self, state):
    #     state = self.bn1(nn.functional.relu(self.conv1(state)))
    #     state = self.bn2(nn.functional.relu(self.conv2(state)))
    #     state = state.view(-1, self.conv_latent_layer)
    #     return state

    def forward(self):
        raise NotImplementedError
        
    def act(self, state, memory):
        # print(state.shape)
        state = torch.from_numpy(state).float().permute(2,0,1).unsqueeze(0).to(device)
        # print(self.debug(state).size())
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        memory.states.append(state.squeeze(0))
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action.item()
    
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.dist_entropy = 0
        
        self.MseLoss = nn.MSELoss()

        self.update_count = 0
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        self.update_count += 1
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        self.dist_entropy = 0
        self.loss_term_1 = 0
        # Optimize policy for K epochs:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr*0.1**(min(int(self.update_count/1000), 11))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 2, 0.5)
        for epoch in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            self.dist_entropy += torch.mean(dist_entropy).cpu().detach().numpy()
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # Neale suggesting setting entropy loss to zero
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.00*dist_entropy
            self.loss_term_1 += torch.mean(-torch.min(surr1, surr2)).cpu().detach().numpy()
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            if epoch == 0:
                print('current lr', self.scheduler.get_last_lr())
            self.scheduler.step()
            self.optimizer.step()
        self.loss_term_1 /= self.K_epochs
        self.dist_entropy /= self.K_epochs
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
def main():
    ############## Hyperparameters ##############
    # env_name = "matsp-v0"
    # creating environment
    step_length = 10
    env = AssignmentWrapper(HighLevelPlanner(ObservationConverter_FOAs(gym.make('matsp-v0')), steps = step_length))
    map_path = '/media/gary/Bridge/MATSP/train/static_2.txt'
    config_path = '/media/gary/Bridge/MATSP/train/config.json'  
    env.reset(custom_board = map_path, config_path = config_path)
    action_dim = len(env.action_list)
    state_dim = env.observation_space_from_wrapper
    action_dim = len(env.action_list)
    render = False
    solved_reward = 100         # stop training if avg_reward > solved_reward
    log_interval = 10         # print avg reward in the interval
    max_episodes = 100000        # max training episodes
    max_steps = 200  #int(env.flag_num*40/env.agent_num/step_length)      # max timesteps in one episode
    n_latent_var = 128         # number of variables in hidden layer
    update_timestep = 2*max_steps*step_length*10      # update policy every n timesteps
    lr = 0.01
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 10               # update policy for K epochs
    eps_clip = 0.2             # clip parameter for PPO
    random_seed = None
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    avg_error = 0
    timestep = 0
    logging_file = []
    # training loop

    entropy = 0
    loss_term_1 = 0
    for i_episode in range(1, max_episodes+1):
        env.seed(random_seed)
        state = env.reset()
        # print(env.env_dict)
        # plt.imshow(env.render())
        # plt.show()
        # plt.imshow(env.nd_map[0]+env.nd_map[1])
        # plt.show()
        optimal_cost = 0 #Multi_TSP_Heuristic(env.env_dict, env.nd_map[0]+env.nd_map[1]).cost
        episode_reward = 0
        for t in range(max_steps):
            timestep += step_length
            
            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            state, reward, done, _ = env.step(action)
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                print('terminal percentage',sum(memory.is_terminals),'/', len(memory.is_terminals))
                print('Average entropy in update:',ppo.dist_entropy)
                print('loss term 1', ppo.loss_term_1)
                entropy = ppo.dist_entropy
                loss_term_1 = ppo.loss_term_1
                memory.clear_memory()
                timestep = 0
            episode_reward += reward
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        avg_error += t*step_length-optimal_cost
        avg_length += t*step_length
        
        # stop training if avg_reward > solved_reward
        if False and running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format('matsp-v0'))
            break
        logging_file.append([i_episode, t,episode_reward, t*step_length-optimal_cost, entropy, loss_term_1])
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            avg_error = int(avg_error/log_interval)
            running_reward = int((running_reward/log_interval))
            flag_num = env.config['elements']['NUM_FLAG']
            agent_num = env.config['elements']['NUM_AGENT']
            # print(flag_num, agent_num)
            np.save('logging_15_FOA_lowlr_f{0}a{1}.npy'.format(flag_num, agent_num), logging_file)
            print('Episode {} \t avg length: {} \t reward: {} \t error: {} \t current_lr: []'.format(i_episode, avg_length, running_reward, avg_error, ))
            running_reward = 0
            avg_length = 0
            avg_error = 0
    torch.save(ppo.policy.state_dict(), './PPO_FOA_{}.pth'.format('matsp-v0'))
if __name__ == '__main__':
    main()