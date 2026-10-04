import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import yaml
import os
import argparse
import random

RUNS_DIR = 'runs'
os.makedirs(RUNS_DIR, exist_ok=True)

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class Agent:
    def __init__(self, parameters_set="flappybirdv0", param_set=None):
        self.parameters_set = param_set if param_set is not None else parameters_set
        with open("parameters.yaml", 'r') as file:
            all_parameters_set = yaml.safe_load(file)
            params = all_parameters_set[self.parameters_set]

        self.alpha = params['alpha']
        self.gamma = params['gamma']

        self.epislon_init = params['epislon_init']
        self.epislon_min = params['epislon_min']
        self.epislon_decay = params['epislon_decay']

        self.replay_memory_size = params['replay_memory_size']
        self.minibatch_size = params['minibatch_size']
        self.mini_batch_size = self.minibatch_size

        self.network_sync_rate = params['network_sync_rate']
        self.reward_threshold = params['reward_threshold']

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        # log files
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.parameters_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.parameters_set}.pt")

    def run(self, is_training=True, render=False):
        env = gym.make(
            "FlappyBird-v0",
            render_mode="human" if render else None,
        )

        # Observation space is continuous (Box), so use shape[0]
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epislon = self.epislon_init

            target_dqn = DQN(num_states, num_actions).to(device)

            # copy the weights from policy network
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
            best_reward = float("-inf")
        else:
            # load best policy
            if os.path.exists(self.MODEL_FILE):
                policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval()
            epislon = 0.0

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0

            while not terminated and episode_reward < self.reward_threshold:
                if is_training and random.random() < epislon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                next_state, reward, terminated, _, _ = env.step(action.item())

                reward = torch.tensor(reward, dtype=torch.float, device=device)
                next_state = torch.tensor(next_state, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    steps += 1

                state = next_state
                episode_reward += reward.item()

            print(f" for episode {episode+1} : total reward value :{episode_reward} & episolon value: {epislon} ")

            if is_training:
                epislon = max(epislon * self.epislon_decay, self.epislon_min)
                if episode_reward > best_reward:
                    log_msg = f"Best Reward {episode_reward} for episode = {episode+1}"
                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                if len(memory) > self.minibatch_size:
                    mini_batch = memory.sample(self.minibatch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                if steps > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # get batch of experiences
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        # calculate target Q-values - if terminations=true => zero
        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]

        # calculate y_pred i.e. Q-value from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # compute loss
        loss = self.loss_fn(current_q, target_q)

        # optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='Hyperparameter set name in parameters.yaml')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(parameters_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
