# Example usage
from src.werewolf_env import WerewolfEnvironment
from src.agent import VanillaLanguageAgent, OmniscientLanguageAgent
from src.strategic_agent import DeductiveLanguageAgent, DiverseDeductiveAgent, StrategicLanguageAgent
from datetime import datetime
import torch
import torch.nn as nn
import argparse
import logging
import numpy as np
import os
import pathlib
import sys

# make env and agents

ROOT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))
env = WerewolfEnvironment()
n_player = env.n_player

# make log dir if not exits
now = datetime.now()
model_dir = f"{ROOT_DIR}/log/{args.name}/{args.model}"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
all_files = os.listdir(model_dir)
n_logs = len(all_files) if "result.json" not in all_files else len(all_files) - 1
log_dir = f"{model_dir}/game{n_logs}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = f"{log_dir}/env.log"
handler = logging.FileHandler(log_file)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger = logging.getLogger("env")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
print(f"The log files can be found at {log_file}.")
agents = [StrategicLanguageAgent("gpt-3.5-turbo-0613", 0.7, f"{log_dir}/player_{i}.log") for i in range(n_player)]


hyperparams = {
    'learning_rate': 5e-4,
    'discount_rate': 0.95,
    'gae_parameter': 0.95,
    'gradient_clipping': 10.0,
    'adam_stepsize': 1e-5,
    'value_loss_coefficient': 1,
    'entropy_coefficient': 0.01,
    'ppo_clipping': 0.2,
    'ppo_epochs': 10,
    'weight_decay_coefficient': 1e-6,
}
import torch
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, embed_size=1536, heads=12):
        super(ActorNetwork, self).__init__()
        self.e_state=0
        self.embed_size = embed_size
        self.heads = heads
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.norm = nn.LayerNorm(embed_size)  # Layer normalization

    def forward(self, combined_embeddings, mask=None):
        N = combined_embeddings.shape[0]
        value_len = key_len = query_len = combined_embeddings.shape[1]

        values = self.values(combined_embeddings).view(N, value_len, self.heads, self.embed_size // self.heads)
        keys = self.keys(combined_embeddings).view(N, key_len, self.heads, self.embed_size // self.heads)
        queries = self.queries(combined_embeddings).view(N, query_len, self.heads, self.embed_size // self.heads)

        values = values.transpose(1, 2)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).transpose(1, 2)
        out = out.contiguous().view(N, query_len, self.embed_size)
        out = self.fc_out(out)
        out = self.norm(out + combined_embeddings)
        # Average Pooling e_player, e_obs
        eh_player = out[0][0]
        eh_obs = out[0][1]
        # Concatenate embeddings along a new dimension to maintain the distinction
        combined_embeddings = torch.cat((eh_player.unsqueeze(0), eh_obs.unsqueeze(0)), dim=0)

        # Average pool across the new dimension
        e_state = combined_embeddings.mean(dim=0).unsqueeze(0).unsqueeze(0)
        print("State representation after concatenation and pooling:", e_state)
        print("Shape of the state representation:", e_state.shape)

        # actions and dot products
        e_actions = out[0, 2:].unsqueeze(0)
        print(f'Actions embeddings Shape: ', e_actions.shape)


        # Flatten e_state to match input dimensions of the critic
        e_state_flat = e_state.view(e_state.size(0), -1)
        self.e_state=e_state_flat

        # Process for action selection
        e_actions = out[:, 2:].unsqueeze(0)  # Actions embeddings
        e_state = out.mean(dim=1).unsqueeze(0)  # State representation
        attended_output, attention_weights = self.scaled_dot_product_attention(e_state, e_actions, e_actions)

        return attended_output, attention_weights

    def scaled_dot_product_attention(self, query, keys, values):
        d_k = query.size(-1)
        scores = torch.matmul(query, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float))
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, values)
        return output, weights
    def get_state(self):
        return self.e_state


actor_network = ActorNetwork()

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Critic, self).__init__()
        # Define the critic network architecture
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output a single value

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value

# Initialize the critic
critic = Critic(input_dim=1536, hidden_dim=512)  # Adjust the dimensions as needed

class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        # Activation and initialization settings
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        # Define layers
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), active_func, nn.LayerNorm(hidden_dim))
        self.fc2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), active_func, nn.LayerNorm(hidden_dim)
            ) for _ in range(layer_N - 1)
        ])
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), active_func, nn.LayerNorm(output_dim))

        # Apply initialization
        self.apply(lambda m: self.init_weights(m, init_method, gain))

    def init_weights(self, module, init_method, gain):
        """ Initialize weights of the neural network """
        if isinstance(module, nn.Linear):
            init_method(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    def forward(self, x):
        x = self.fc1(x)
        for layer in self.fc2:
            x = layer(x)
        x = self.fc3(x)
        return x


player_encoder = MLPLayer(input_dim=253, output_dim=1536, hidden_dim=512, layer_N=3, use_orthogonal=True, use_ReLU=True)

class MultiAgentPPO:
    def __init__(self, num_agents, input_dim, hidden_dim, action_dim, hyperparams):
        self.actors = [Actor(input_dim, hidden_dim, action_dim) for _ in range(num_agents)]
        self.critic = Critic(num_agents * input_dim, hidden_dim)  # Centralized critic
        self.optimizers = [optim.Adam(actor.parameters(), lr=hyperparams['learning_rate']) for actor in self.actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=hyperparams['learning_rate'])
        self.hyperparams = hyperparams
        self.num_agents = num_agents

    def act(self, states):
        actions = []
        log_probs = []
        for actor, state in zip(self.actors, states):
            probs = actor(state)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            actions.append(action)
            log_probs.append(log_prob)
        return actions, log_probs

    def evaluate(self, states, actions):
        state_tensor = torch.cat(states, dim=1)  # Concatenate states for centralized critic
        values = self.critic(state_tensor)
        action_log_probs = []
        dist_entropies = []
        for actor, state, action in zip(self.actors, states, actions):
            action_probs = actor(state)
            dist = Categorical(action_probs)
            action_log_probs.append(dist.log_prob(action))
            dist_entropies.append(dist.entropy())
        return torch.cat(action_log_probs), values, torch.cat(dist_entropies)

def train_mappo(agents, env, episodes, steps_per_episode):
    for episode in range(episodes):
        states = env.reset()
        memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': []}
        for step in range(steps_per_episode):
            actions, log_probs = agents.act([torch.from_numpy(state).float() for state in states])
            next_states, rewards, dones, _ = env.step([action.item() for action in actions])

            memory['states'].extend(states)
            memory['actions'].extend(actions)
            memory['log_probs'].extend(log_probs)
            memory['rewards'].extend(rewards)
            memory['dones'].extend(dones)
            values = [agents.critic(torch.cat([torch.from_numpy(state).float() for state in states], dim=1))]
            memory['values'].extend(values)

            states = next_states

            if all(dones):
                break

        # Convert lists to torch tensors
        states = torch.stack(memory['states'])
        actions = torch.stack(memory['actions'])
        log_probs = torch.stack(memory['log_probs'])
        rewards = torch.tensor(memory['rewards'], dtype=torch.float32)
        dones = torch.tensor(memory['dones'], dtype=torch.float32)
        values = torch.cat(memory['values'])

        # Calculate advantages and returns
        returns = []
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + agents.hyperparams['discount_rate'] * values[i + 1] * dones[i] - values[i]
            gae = delta + agents.hyperparams['discount_rate'] * agents.hyperparams['gae_parameter'] * dones[i] * gae
            returns.insert(0, gae + values[i])
            advantages.insert(0, gae)
        returns = torch.tensor(returns)
        advantages = torch.tensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy and value updates
        for epoch in range(agents.hyperparams['ppo_epochs']):
            log_probs_new, values_new, entropies = agents.evaluate([states[i] for i in range(agents.num_agents)], actions)
            ratio = torch.exp(log_probs_new - log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - agents.hyperparams['ppo_clipping'], 1.0 + agents.hyperparams['ppo_clipping']) * advantages
            policy_loss = -torch.min(surr1, surr2) - agents.hyperparams['entropy_coefficient'] * entropies.mean()
            value_loss = F.mse_loss(returns, values_new)
            loss = policy_loss + agents.hyperparams['value_loss_coefficient'] * value_loss

            # Update actor and critic
            for optimizer in agents.optimizers:
                optimizer.zero_grad()
            agents.critic_optimizer.zero_grad()
            loss.backward()
            for optimizer in agents.optimizers:
                optimizer.step()
            agents.critic_optimizer.step()

        print(f"Episode {episode+1} complete with final reward: {rewards[-1]}")

# Usage example remains the same as previously defined.

