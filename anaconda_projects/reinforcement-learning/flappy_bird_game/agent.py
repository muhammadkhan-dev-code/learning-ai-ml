import torch  
import torch.nn as nn
import torch.optim as optim
import itertools
import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import yaml 



if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class Agent:
    def __init__(self,parameters_set):
        self.parameters_set= parameters_set
        with open("parameters.yaml" , 'r') as file:
            all_parameters_set=yaml.safe_load(file)
            params=all_parameters_set[parameters_set]

        self.alpha=params['alpha']
        self.gamma=params['gamma']

        self.episoln_init=params['epislon_init']
        self.epislon_min=params['epislon_min']
        self.epislon_decay=params['epislon_decay']

        self.replay_memory_size=params['replay_memory_size']
        self.minibatch_size=params['minibatch_size']

        self.network_sync_rate=params['network_sync_rate']
        self.reward_threshold=params['reward_threshold']


        self.loss_func=nn.MSELoss()
        self.optimizer =None




    def run(is_training=True, render=False):
        env = gym.make(
            "FlappyBird-v0",
            render_mode="human" if render else None,
        )

        # Fixed: FlappyBird observation space is continuous (Box), so use shape[0] instead of .n
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epislon= self.epislon_init



        for episode in itertools.count():

           state, _ = env.reset()
        #    state into tensors

           state=torch.tnsor(state,dtype=torch.float,device=device)


           terminated = False
           episode_rewards = 0  # Fixed variable name from episode_rewards to match loop body

           while not terminated:
               
               if is_training and random.random()< epislon:
                    action = env.action_space.sample()   # Next action:  (feed the observation to your agent here)
                    action=torch.tnsor(action,dtype=torch.long,device=device)
               else:
                   with torch.no_grad ():
                        action=policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() # exploitation
                
               # Processing: terminated = done
               next_state, reward, terminated, _, _ = env.step(action.item())

            #    create tensors\
               reward=torch.tnsor(reward,dtype=torch.float,device=device)
               next_state=torch.tnsor(next_state,dtype=torch.float,device=device)
                

               if is_training:
                   # Fixed: replaced undefined new_state with next_state
                   memory.append((state, action, next_state, reward, terminated))
                   # Checking if the player is still alive
               # state reinitialized with next_state (fixed new_state typo)
               state = next_state
               # Fixed: changed rewards to reward and matching episode_reward
               episode_rewards += reward
           print(f" for episode {episode+1} : total reward value :{episode_rewards} & episolon value: {epislon} ")

           # env.close() indefinitie run our code just solve this not added from your side

        #    episolon decay
           epislon = max(epislon * self.epislon_decay,self.epislon_min)
