import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Use from tqdm.notebook import tqdm if in a Jupyter notebook
import time
import random
from collections import deque
import copy # For deep copying models/q-tables

import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle # Keep for potential future use if needed
from matplotlib.lines import Line2D

# PyTorch Imports for DQN
import torch
import torch.nn as nn
import torch.optim as optim

# Set LaTeX font to Times New Roman (Optional, comment out if not needed or causing issues)
# try:
#     plt.rcParams['font.family'] = 'serif'
#     plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
#     plt.rcParams['mathtext.fontset'] = 'stix' # Or 'cm' for Computer Modern
# except Exception as e:
#     print(f"Could not set Times New Roman font: {e}")


# --- Environment Availability Check and Dummy Definitions ---
_REAL_ENV_AVAILABLE = False
_PCGYM_AVAILABLE = False
try:
    from pcgym import make_env
    _PCGYM_AVAILABLE = True
    _REAL_ENV_AVAILABLE = True # Assume if pcgym imports, real env can be made
    print("pcgym imported successfully.")
except ImportError as e:
    print(f"Error importing pcgym: {e}")
    print("Please ensure 'pcgym' is installed and in the Python path.")
    print("DQN training and 'real' Q-learning environment will be skipped.")
    print("Q-learning will use a dummy environment.")

class DummyEnv:
    def __init__(self, n_actions=7, grid_size=(10,10)):
        self._n_actions = n_actions
        self.grid_size = grid_size
        self.observation_space = type('space', (object,), {'shape': (2,)})() # Discrete state [idx_ca, idx_temp]
        self.action_space = type('space', (object,), {'n': self._n_actions})() # Discrete actions
        self.state = np.array([0,0]) # Current discrete state
        self._max_episode_steps = 50
        self.N = self._max_episode_steps # For compatibility with env.N

        # For providing mock continuous values if asked by a wrapper
        self.mock_ca_range = (0.7, 0.9)
        self.mock_temp_range = (315., 335.)
        self.mock_goal_ca = 0.85

    def _get_mock_continuous_state(self, discrete_state):
        ca_val = self.mock_ca_range[0] + (discrete_state[0] / (self.grid_size[0]-1)) * (self.mock_ca_range[1] - self.mock_ca_range[0])
        temp_val = self.mock_temp_range[0] + (discrete_state[1] / (self.grid_size[1]-1)) * (self.mock_temp_range[1] - self.mock_temp_range[0])
        return np.array([ca_val, temp_val, self.mock_goal_ca]) # [Ca, Temp, Ca_SP]

    def reset(self, seed=None):
        if seed is not None: np.random.seed(seed)
        self.state = np.array([np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1])])
        info = {'continuous_state': self._get_mock_continuous_state(self.state)}
        return self.state.copy(), info # Return discrete state

    def step(self, action_idx): # Expects a discrete action index
        action_effect = 0
        if self._n_actions > 1:
            if action_idx < self._n_actions / 2:
                action_effect = -1
            elif action_idx >= self._n_actions / 2 :
                action_effect = 1
                if self._n_actions % 2 == 1 and action_idx == self._n_actions // 2:
                    action_effect = 0
        self.state[1] = np.clip(self.state[1] + action_effect, 0, self.grid_size[1] - 1)
        self.state[0] = np.clip(self.state[0] + np.random.randint(-1, 2), 0, self.grid_size[0] - 1)
        target_discrete_ca = self.grid_size[0] - 2
        target_discrete_temp = self.grid_size[1] // 2
        reward = -np.sqrt((self.state[0] - target_discrete_ca)**2 + (self.state[1] - target_discrete_temp)**2)
        terminated = False
        truncated = False
        info = {'continuous_state': self._get_mock_continuous_state(self.state)}
        return self.state.copy(), reward, terminated, truncated, info

    def close(self):
        pass
    class MockObservationSpace:
        def __init__(self, low, high):
            self.low = np.array(low, dtype=np.float32)
            self.high = np.array(high, dtype=np.float32)
            self.shape = self.low.shape
    class MockActionSpace:
        def __init__(self, low, high):
            self.low = np.array(low)
            self.high = np.array(high)

if not _PCGYM_AVAILABLE:
    def make_env(params):
        print("WARNING: pcgym not available. Using DUMMY make_env, returning DummyEnv.")
        n_dummy_actions = params.get('n_actions_for_dummy', 7)
        dummy_grid_size = params.get('grid_size_for_dummy', (10,10))
        mock_obs_low = [0.7, 315., 0.8]; mock_obs_high = [0.9, 335., 0.9]
        mock_act_low = [295.]; mock_act_high = [302.]
        dummy = DummyEnv(n_actions=n_dummy_actions, grid_size=dummy_grid_size)
        dummy.observation_space = DummyEnv.MockObservationSpace(mock_obs_low, mock_obs_high)
        dummy.action_space = DummyEnv.MockActionSpace(mock_act_low, mock_act_high)
        return dummy

class DiscreteReactorWrapper:
    def __init__(self, continuous_env, grid_size, goal_ca_continuous, num_discrete_actions_q_learning):
        self.continuous_env = continuous_env
        self.grid_size = grid_size
        self.goal_ca_continuous = goal_ca_continuous
        self.n_actions = num_discrete_actions_q_learning
        self.ca_min = self.continuous_env.observation_space.low[0]
        self.ca_max = self.continuous_env.observation_space.high[0]
        self.temp_min = self.continuous_env.observation_space.low[1]
        self.temp_max = self.continuous_env.observation_space.high[1]
        self.ca_bins = np.linspace(self.ca_min, self.ca_max, self.grid_size[0] + 1)
        self.temp_bins = np.linspace(self.temp_min, self.temp_max, self.grid_size[1] + 1)
        self.x0_continuous_from_reset = None
        if hasattr(self.continuous_env, 'x0'):
            self.initial_continuous_state_nominal = self.continuous_env.x0[:2].copy()
        else:
            self.initial_continuous_state_nominal = np.array([ (self.ca_bins[0]+self.ca_bins[-1])/2,
                                                               (self.temp_bins[0]+self.temp_bins[-1])/2 ])
        self.action_low_cont = self.continuous_env.action_space.low[0]
        self.action_high_cont = self.continuous_env.action_space.high[0]
        if self.n_actions > 1:
            self.q_learning_discrete_to_continuous_actions = np.linspace(self.action_low_cont, self.action_high_cont, self.n_actions)
        elif self.n_actions == 1:
            self.q_learning_discrete_to_continuous_actions = np.array([(self.action_low_cont + self.action_high_cont) / 2])
        else:
            self.q_learning_discrete_to_continuous_actions = np.array([])
    def _discretize_state(self, continuous_state_vector):
        ca_val = continuous_state_vector[0]; temp_val = continuous_state_vector[1]
        ca_idx = np.digitize(ca_val, self.ca_bins) - 1
        temp_idx = np.digitize(temp_val, self.temp_bins) - 1
        ca_idx = np.clip(ca_idx, 0, self.grid_size[0] - 1)
        temp_idx = np.clip(temp_idx, 0, self.grid_size[1] - 1)
        return np.array([ca_idx, temp_idx], dtype=int)
    def _map_discrete_action_to_continuous(self, discrete_action_idx):
        if isinstance(self.continuous_env, DummyEnv): return discrete_action_idx
        return np.array([self.q_learning_discrete_to_continuous_actions[discrete_action_idx]])
    def reset(self, seed=None):
        if isinstance(self.continuous_env, DummyEnv):
            state_discrete, info = self.continuous_env.reset(seed=seed)
            self.x0_continuous_from_reset = info.get('continuous_state', np.array([self.ca_min, self.temp_min, self.goal_ca_continuous]))[:2]
            return state_discrete, info
        continuous_obs, info = self.continuous_env.reset(seed=seed)
        self.x0_continuous_from_reset = continuous_obs[:2].copy()
        state_discrete = self._discretize_state(continuous_obs)
        return state_discrete, info
    def step(self, discrete_action_idx):
        if isinstance(self.continuous_env, DummyEnv):
            next_state_discrete, reward, terminated, truncated, info = self.continuous_env.step(discrete_action_idx)
            return next_state_discrete, reward, terminated, truncated, info
        continuous_action_val = self._map_discrete_action_to_continuous(discrete_action_idx)
        next_continuous_obs, reward, terminated, truncated, info_cont = self.continuous_env.step(continuous_action_val)
        next_state_discrete = self._discretize_state(next_continuous_obs)
        return next_state_discrete, reward, terminated, truncated, info_cont
    def close(self): self.continuous_env.close()
    @property
    def N(self): return self.continuous_env.N

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim1=128, hidden_dim2=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim1); self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, action_size)
    def forward(self, state):
        x = torch.relu(self.fc1(state)); x = torch.relu(self.fc2(x)); return self.fc3(x)

class DQN_Agent:
    def __init__(self, state_size, num_discrete_actions, continuous_action_bounds, device="cpu"):
        self.state_size = state_size; self.num_discrete_actions = num_discrete_actions
        self.memory = deque(maxlen=20000); self.gamma = 0.99; self.epsilon = 1.0
        self.epsilon_min = 0.01; self.epsilon_decay = 0.999; self.learning_rate = 0.0005
        self.batch_size = 64; self.device = device; self.update_target_freq = 10
        self.train_step_counter = 0
        self.action_low_cont = continuous_action_bounds['low'][0]
        self.action_high_cont = continuous_action_bounds['high'][0]
        if self.num_discrete_actions > 1:
            self.discrete_actions_map_to_continuous = np.linspace(self.action_low_cont, self.action_high_cont, self.num_discrete_actions)
        elif self.num_discrete_actions == 1:
             self.discrete_actions_map_to_continuous = np.array([(self.action_low_cont + self.action_high_cont)/2])
        else: self.discrete_actions_map_to_continuous = np.array([])
        self.q_network = QNetwork(state_size, num_discrete_actions).to(device)
        self.target_network = QNetwork(state_size, num_discrete_actions).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.update_target_network()
    def update_target_network(self): self.target_network.load_state_dict(self.q_network.state_dict())
    def remember(self, state, action_idx, reward, next_state, done): self.memory.append((state, action_idx, reward, next_state, done))
    def _get_discrete_action_index(self, continuous_action_value):
        if not self.discrete_actions_map_to_continuous.any(): return 0
        return np.argmin(np.abs(self.discrete_actions_map_to_continuous - continuous_action_value))
    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.num_discrete_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.q_network.eval()
            with torch.no_grad(): action_values = self.q_network(state_tensor)
            self.q_network.train(); action_idx = torch.argmax(action_values).item()
        continuous_action_val = np.array([self.discrete_actions_map_to_continuous[action_idx]])
        return continuous_action_val, action_idx
    def get_q_values(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.q_network.eval()
        with torch.no_grad(): q_values = self.q_network(state_tensor)
        self.q_network.train(); return q_values.cpu().numpy()[0]
    def train_batch(self):
        if len(self.memory) < self.batch_size: return 0.0
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([exp[0] for exp in minibatch]); actions_idx = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch]); next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_idx_tensor = torch.LongTensor(actions_idx).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        current_q_vals = self.q_network(states_tensor).gather(1, actions_idx_tensor.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions_online = self.q_network(next_states_tensor).argmax(dim=1)
            next_q_vals_target = self.target_network(next_states_tensor).gather(1, next_actions_online.unsqueeze(1)).squeeze(1)
            target_q_vals = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_vals_target
        loss = nn.MSELoss()(current_q_vals, target_q_vals)
        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay
        self.train_step_counter += 1
        if self.train_step_counter % self.update_target_freq == 0: self.update_target_network()
        return loss.item()
    def save_weights(self, path): torch.save(self.q_network.state_dict(), path)
    def load_weights(self, path):
        self.q_network.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target_network()

class QLearningAgent:
    def __init__(self, state_dim_shape, action_dim_size, learning_rate=0.1, discount_factor=0.99,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        self.state_dim_shape = state_dim_shape; self.action_dim_size = action_dim_size
        self.lr = learning_rate; self.gamma = discount_factor; self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay; self.min_epsilon = min_exploration_rate
        self.q_table = np.zeros(state_dim_shape + (action_dim_size,))
        print(f"Initialized Q-learning table with shape: {self.q_table.shape}")
    def choose_action(self, state_tuple):
        if np.random.random() < self.epsilon: return np.random.randint(0, self.action_dim_size)
        else:
            q_values_for_state = self.q_table[state_tuple]
            return np.random.choice(np.flatnonzero(q_values_for_state == np.max(q_values_for_state)))
    def update(self, state_tuple, action_idx, reward, next_state_tuple, done):
        current_q = self.q_table[state_tuple][action_idx]
        best_next_q = np.max(self.q_table[next_state_tuple]) if not done else 0.0
        target_q = reward + self.gamma * best_next_q
        self.q_table[state_tuple][action_idx] = current_q + self.lr * (target_q - current_q)
    def decay_epsilon(self): self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

def create_continuous_reactor_env_for_dqn():
    if not _PCGYM_AVAILABLE: return None
    try:
        nsteps = 60; T = 26.0; goal_ca_cont = 0.86
        SP = {'Ca': [goal_ca_cont for _ in range(int(nsteps))]}
        action_space_cont = {'low': np.array([295.]), 'high': np.array([302.])}
        observation_space_cont = {
            'low': np.array([0.7, 315., 0.8], dtype=np.float32),
            'high': np.array([0.9, 335., 0.9], dtype=np.float32),
        }
        r_scale = {'Ca': 1e3}; initial_x0 = np.array([0.72, 328., goal_ca_cont])
        env_params = {'N': nsteps, 'tsim': T, 'SP': SP, 'o_space': observation_space_cont,
            'a_space': action_space_cont, 'x0': initial_x0, 'model': 'cstr', 'r_scale': r_scale,
            'normalise_a': True, 'normalise_o': True, 'noise': True,
            'integration_method': 'casadi', 'noise_percentage': 0.001}
        cont_env = make_env(env_params)
        cont_env.goal_ca_continuous = goal_ca_cont
        print("Continuous CSTR environment for DQN created successfully.")
        return cont_env
    except Exception as e: print(f"Error creating continuous CSTR environment for DQN: {e}"); return None

def create_q_learning_env_and_agent(num_q_actions, grid_size_q, goal_ca_q):
    base_env_for_q_wrapper = None
    if _PCGYM_AVAILABLE:
        print("Attempting to create Q-learning environment with REAL pcgym base.")
        try:
            nsteps = 25; T = 26.0; SP_q = {'Ca': [goal_ca_q for _ in range(int(nsteps))]}
            action_space_q = {'low': np.array([295.]), 'high': np.array([302.])}
            observation_space_q = {'low': np.array([0.7, 315., 0.8], dtype=np.float32),
                                   'high': np.array([0.9, 335., 0.9], dtype=np.float32)}
            r_scale_q = {'Ca': 1e3}; initial_x0_q = np.array([0.71, 326., goal_ca_q])
            env_params_q = {'N': nsteps, 'tsim': T, 'SP': SP_q, 'o_space': observation_space_q,
                'a_space': action_space_q, 'x0': initial_x0_q, 'model': 'cstr', 'r_scale': r_scale_q,
                'normalise_a': False, 'normalise_o': False, 'noise': True,
                'integration_method': 'casadi', 'noise_percentage': 0.001}
            base_env_for_q_wrapper = make_env(env_params_q)
        except Exception as e: print(f"Failed to create real pcgym base for Q-learning: {e}."); base_env_for_q_wrapper = None
    if base_env_for_q_wrapper is None:
        print("Creating Q-learning environment with DUMMY base.")
        base_env_for_q_wrapper = make_env({'n_actions_for_dummy': num_q_actions, 'grid_size_for_dummy': grid_size_q})
        base_env_for_q_wrapper.goal_ca_continuous = goal_ca_q
    wrapped_q_env = DiscreteReactorWrapper(base_env_for_q_wrapper, grid_size_q, goal_ca_q, num_q_actions)
    q_agent = QLearningAgent(state_dim_shape=grid_size_q, action_dim_size=num_q_actions,
                             learning_rate=0.1, exploration_decay=0.998, min_exploration_rate=0.01)
    return wrapped_q_env, q_agent

def train_dqn_agent(dqn_env, dqn_agent, episodes, dqn_action_size):
    if dqn_env is None or dqn_agent is None: return [], {}
    print(f"Starting DQN training for {episodes} episodes...")
    dqn_scores = []; dqn_q_func_snapshots = {'early': None, 'middle': None, 'late': None}
    ep_early = max(1, int(episodes * 0.05)); ep_middle = max(ep_early + 1, int(episodes * 0.5))
    snapshot_agent = DQN_Agent(dqn_agent.state_size, dqn_action_size,
                               {'low': np.array([dqn_agent.action_low_cont]), 'high': np.array([dqn_agent.action_high_cont])},
                               device=dqn_agent.device)
    progress_bar = tqdm(range(episodes), desc="DQN Training")
    for e in progress_bar:
        state_cont, _ = dqn_env.reset()

        total_reward_ep = 0
        for _ in range(dqn_env.N):
            continuous_action_val, action_idx = dqn_agent.act(state_cont, training=True)
            next_state_cont, reward, terminated, truncated, _ = dqn_env.step(continuous_action_val)

            done = terminated or truncated
            dqn_agent.remember(state_cont, action_idx, reward, next_state_cont, done)
            state_cont = next_state_cont; total_reward_ep += reward
            loss = dqn_agent.train_batch()
            if done: break
        dqn_scores.append(total_reward_ep)
        progress_bar.set_postfix({'R': f"{total_reward_ep:.1f}", 'Eps': f"{dqn_agent.epsilon:.2f}", 'L': f"{loss:.3f}"})
        if e == ep_early:
            snapshot_agent.q_network.load_state_dict(dqn_agent.q_network.state_dict())
            dqn_q_func_snapshots['early'] = copy.deepcopy(snapshot_agent.q_network.state_dict())
        elif e == ep_middle:
            snapshot_agent.q_network.load_state_dict(dqn_agent.q_network.state_dict())
            dqn_q_func_snapshots['middle'] = copy.deepcopy(snapshot_agent.q_network.state_dict())
    snapshot_agent.q_network.load_state_dict(dqn_agent.q_network.state_dict())
    dqn_q_func_snapshots['late'] = copy.deepcopy(snapshot_agent.q_network.state_dict())
    if dqn_q_func_snapshots['early'] is None and dqn_q_func_snapshots['late'] is not None:
        dqn_q_func_snapshots['early'] = copy.deepcopy(dqn_q_func_snapshots['late'])
    if dqn_q_func_snapshots['middle'] is None and dqn_q_func_snapshots['late'] is not None:
        dqn_q_func_snapshots['middle'] = copy.deepcopy(dqn_q_func_snapshots['late'])
    return dqn_scores, dqn_q_func_snapshots

def train_q_learning_agent(q_env, q_agent, episodes):
    print(f"Starting Q-learning training for {episodes} episodes...")
    q_learning_rewards = []; q_tables_history = {'early': None, 'middle': None, 'late': None}
    ep_early = max(1, int(episodes * 0.05)); ep_middle = max(ep_early + 1, int(episodes * 0.5))
    progress_bar = tqdm(range(episodes), desc="Q-learning Training")
    for episode in progress_bar:
        state_discrete, _ = q_env.reset(); state_tuple = tuple(state_discrete)
        total_reward_ep = 0
        for _ in range(q_env.N):
            action_idx = q_agent.choose_action(state_tuple)
            next_state_discrete, reward, terminated, truncated, _ = q_env.step(action_idx)
            next_state_tuple = tuple(next_state_discrete); done = terminated or truncated
            q_agent.update(state_tuple, action_idx, reward, next_state_tuple, done)
            state_tuple = next_state_tuple; total_reward_ep += reward
            if done: break
        q_agent.decay_epsilon(); q_learning_rewards.append(total_reward_ep)
        progress_bar.set_postfix({'R': f"{total_reward_ep:.1f}", 'Eps': f"{q_agent.epsilon:.2f}"})
        if episode == ep_early: q_tables_history['early'] = q_agent.q_table.copy()
        elif episode == ep_middle: q_tables_history['middle'] = q_agent.q_table.copy()
    q_tables_history['late'] = q_agent.q_table.copy()
    if q_tables_history['early'] is None and q_tables_history['late'] is not None:
         q_tables_history['early'] = np.zeros_like(q_tables_history['late'])
    if q_tables_history['middle'] is None and q_tables_history['late'] is not None:
         q_tables_history['middle'] = q_tables_history['late'].copy() # Should be an earlier snapshot or zeros if truly middle wasn't hit
    return q_learning_rewards, q_tables_history


# --- Visualization Function for Value Function V(s) ---
def visualize_value_function_heatmap(value_func_data, ca_edges_vis, temp_edges_vis, goal_ca_val_vis,
                                     ax, title="V(s)", is_first_col=False, is_last_row=False,
                                     is_dqn_plot=False, unvisited_mask=None): # Added unvisited_mask
    if value_func_data is None:
        ax.text(0.5, 0.5, "V(s) data not available", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        return ax, []

    # Make a copy of the colormap to modify it locally for this subplot
    cmap_val = plt.cm.PiYG.copy()
    value_display_data = value_func_data # Default to original data

    if unvisited_mask is not None and np.any(unvisited_mask):
        # If there's an unvisited mask and it actually masks something
        value_display_data = np.ma.array(value_func_data, mask=unvisited_mask)
        cmap_val.set_bad(color='lightgrey') # Color for masked (unvisited/initial Q=0) states
    
    # Calculate v_min, v_max from the *unmasked* part of value_display_data
    if isinstance(value_display_data, np.ma.MaskedArray):
        compressed_data = value_display_data.compressed() # 1D array of unmasked values
        v_finite = compressed_data[np.isfinite(compressed_data)]
    else:
        v_finite = value_display_data[np.isfinite(value_display_data)]

    if v_finite.size > 0:
        v_min, v_max = np.min(v_finite), np.max(v_finite)
    else: # All data is masked or non-finite
        v_min, v_max = -1.0, 1.0 # Default range if no valid data points
    if abs(v_min - v_max) < 1e-9: v_min -= 0.5; v_max += 0.5

    norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
    interpolation_method = 'bicubic' if is_dqn_plot else 'nearest'

    plot_extent = [temp_edges_vis[0], temp_edges_vis[-1], ca_edges_vis[0], ca_edges_vis[-1]]
    # Use value_display_data for plotting
    im = ax.imshow(value_display_data, extent=plot_extent, aspect='auto', cmap=cmap_val, origin='lower',
                   interpolation=interpolation_method, norm=norm, zorder=1)

    goal_line = ax.axhline(y=goal_ca_val_vis, color='red', linestyle='--', linewidth=1.2,
                           label=f'Goal Ca ({goal_ca_val_vis:.2f})', zorder=10)
    ax.set_xlim(temp_edges_vis[0], temp_edges_vis[-1])
    ax.set_ylim(ca_edges_vis[0], ca_edges_vis[-1])

    num_x_ticks = min(6, len(temp_edges_vis)); num_y_ticks = min(6, len(ca_edges_vis))
    ax.set_xticks(np.linspace(temp_edges_vis[0], temp_edges_vis[-1], num_x_ticks))
    ax.set_yticks(np.linspace(ca_edges_vis[0], ca_edges_vis[-1], num_y_ticks))

    ax.set_xticklabels([f"{t:.0f}" for t in ax.get_xticks()], rotation=30, ha='right', fontsize=6)
    ax.set_yticklabels([f"{ca:.2f}" for ca in ax.get_yticks()], fontsize=6)

    if not is_dqn_plot: # Only draw grid lines for non-DQN (i.e., Q-learning) plots
        ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.2, alpha=0.4, zorder=0)

    ax.set_title(title, fontsize=10, pad=5)

    if is_first_col: ax.set_ylabel('Concentration Ca (mol/L)', fontsize=8)
    if is_last_row: ax.set_xlabel('Reactor Temp. (K)', fontsize=8)

    cbar = plt.colorbar(im, ax=ax, label='State Value V(s)', fraction=0.046, pad=0.04, aspect=15)
    cbar.ax.tick_params(labelsize=6); cbar.set_label('State Value V(s)', size=7)

    return ax, [goal_line]


def get_dqn_q_grid(dqn_agent_eval_mode, ca_edges_grid, temp_edges_grid, fixed_ca_sp_for_state, dqn_continuous_env_for_norm_info):
    num_ca_cells = len(ca_edges_grid) - 1; num_temp_cells = len(temp_edges_grid) - 1
    dqn_action_s = dqn_agent_eval_mode.num_discrete_actions
    q_grid_dqn = np.zeros((num_ca_cells, num_temp_cells, dqn_action_s))
    ca_centers = (ca_edges_grid[:-1] + ca_edges_grid[1:]) / 2
    temp_centers = (temp_edges_grid[:-1] + temp_edges_grid[1:]) / 2

    # Use observation space from the passed environment for normalization info
    observation_space_cont = {
            'low': np.array([0.7, 315., 0.8], dtype=np.float32),
            'high': np.array([0.9, 335., 0.9], dtype=np.float32),
        }
    for r_idx, ca_center_val in enumerate(ca_centers):
        for c_idx, temp_center_val in enumerate(temp_centers):
            continuous_state_unnormalized = np.array([ca_center_val, temp_center_val, fixed_ca_sp_for_state], dtype=np.float32)
            
            state_to_feed_dqn = 2 * (continuous_state_unnormalized - observation_space_cont['low']) / (observation_space_cont['high'] - observation_space_cont['low']) - 1


            q_values_for_state = dqn_agent_eval_mode.get_q_values(state_to_feed_dqn)
            q_grid_dqn[r_idx, c_idx, :] = q_values_for_state
    return q_grid_dqn

def plot_reward_curve(scores, title, algo_name):
    plt.figure(figsize=(8, 4))
    plt.plot(scores, alpha=0.7, label=f'Episode Reward ({algo_name})')
    window = max(1, len(scores) // 20)
    if len(scores) >= window and window > 0:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window - 1, len(scores)), moving_avg, color='red', label=f'{window}-Ep Moving Avg')
    plt.title(title, fontsize=12); plt.xlabel('Episode', fontsize=10); plt.ylabel('Total Reward', fontsize=10)
    plt.legend(fontsize=8); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout(); plt.show()


# --- Main Script ---
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    VIS_GRID_SIZE = (10, 10); GOAL_CA_SHARED = 0.86
    DQN_EPISODES = 100 if _PCGYM_AVAILABLE else 0 
    DQN_ACTION_SIZE = 21
    QL_EPISODES = 300 ; QL_ACTION_SIZE = 7; QL_GRID_SIZE = VIS_GRID_SIZE

    continuous_env_for_dqn = None; dqn_agent = None
    dqn_scores_history = []; dqn_q_func_state_dicts = {'early': None, 'middle': None, 'late': None}
    if _PCGYM_AVAILABLE and DQN_EPISODES > 0:
        continuous_env_for_dqn = create_continuous_reactor_env_for_dqn()
        if continuous_env_for_dqn:
            state_size_dqn = continuous_env_for_dqn.observation_space.shape[0]
            action_bounds_dqn = {'low': continuous_env_for_dqn.action_space.low, 'high': continuous_env_for_dqn.action_space.high}
            dqn_agent = DQN_Agent(state_size_dqn, DQN_ACTION_SIZE, action_bounds_dqn, device=DEVICE)
            dqn_scores_history, dqn_q_func_state_dicts = train_dqn_agent(
                continuous_env_for_dqn, dqn_agent, DQN_EPISODES, DQN_ACTION_SIZE)
            plot_reward_curve(dqn_scores_history, "DQN Training Rewards", "DQN")
    else: print("Skipping DQN training (pcgym not available or DQN_EPISODES is 0).")

    q_learning_env_wrapped, q_learning_agent = create_q_learning_env_and_agent(
        num_q_actions=QL_ACTION_SIZE, grid_size_q=QL_GRID_SIZE, goal_ca_q=GOAL_CA_SHARED)
    q_learning_rewards_history, q_tables_history_ql = train_q_learning_agent(
        q_learning_env_wrapped, q_learning_agent, QL_EPISODES)
    plot_reward_curve(q_learning_rewards_history, "Q-learning Training Rewards", "Q-learning")

    ca_edges_vis = q_learning_env_wrapped.ca_bins
    temp_edges_vis = q_learning_env_wrapped.temp_bins
    goal_ca_plot_val = q_learning_env_wrapped.goal_ca_continuous
    if q_learning_env_wrapped.x0_continuous_from_reset is not None:
        initial_cont_state_for_plot = q_learning_env_wrapped.x0_continuous_from_reset[:2]
    else: initial_cont_state_for_plot = q_learning_env_wrapped.initial_continuous_state_nominal[:2]
    print(f"Initial state for plot marker (Ca, Temp): {initial_cont_state_for_plot}")

    # --- Prepare Value Functions for Visualization ---
    dqn_v_grids_for_plot = {'early': None, 'middle': None, 'late': None}
    ql_v_tables_for_plot = {'early': None, 'middle': None, 'late': None}
    ql_unvisited_masks = {'early': None, 'middle': None, 'late': None} # For Q-learning unvisited states

    print("Generating DQN Q-function grids and V-function grids for visualization...")
    if _PCGYM_AVAILABLE and dqn_agent is not None and DQN_EPISODES > 0 and continuous_env_for_dqn is not None:
        eval_dqn_agent = DQN_Agent(dqn_agent.state_size, DQN_ACTION_SIZE,
                                   {'low': np.array([dqn_agent.action_low_cont]), 'high': np.array([dqn_agent.action_high_cont])},
                                   device=DEVICE)
        eval_dqn_agent.q_network.eval()
        for stage in ['early', 'middle', 'late']:
            if dqn_q_func_state_dicts.get(stage) is not None:
                eval_dqn_agent.q_network.load_state_dict(dqn_q_func_state_dicts[stage])
                temp_q_grid_dqn = get_dqn_q_grid(eval_dqn_agent, ca_edges_vis, temp_edges_vis,
                                                 fixed_ca_sp_for_state=GOAL_CA_SHARED,
                                                 dqn_continuous_env_for_norm_info=continuous_env_for_dqn)
                if temp_q_grid_dqn is not None:
                    dqn_v_grids_for_plot[stage] = np.max(temp_q_grid_dqn, axis=2) 
            else: print(f"  DQN weights for stage: {stage} not available.")
    else: print("DQN Q/V-function grids cannot be generated (pcgym not available, DQN not trained, or continuous_env_for_dqn is None).")

    print("Calculating Q-learning V-function tables and unvisited masks for visualization...")
    for stage in ['early', 'middle', 'late']:
        if q_tables_history_ql.get(stage) is not None:
            current_q_table = q_tables_history_ql[stage]
            ql_v_tables_for_plot[stage] = np.max(current_q_table, axis=2) # V(s) = max_a Q(s,a)
            # Mask is True if all Q-values for that state are 0 (initial value)
            ql_unvisited_masks[stage] = np.all(current_q_table == 0, axis=2)
        else: 
            print(f" QL Q-table for stage {stage} not available.")
            # ql_v_tables_for_plot[stage] remains None
            # ql_unvisited_masks[stage] remains None

    # --- Create the 2x3 Plot for V(s) ---
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=False)
    plt.subplots_adjust(hspace=0.35, wspace=0.3, top=0.92, bottom=0.1, left=0.07, right=0.95)

    plot_titles_map_v = {
        (0,0): "DQN V(s) - Early", (0,1): "DQN V(s) - Middle", (0,2): "DQN V(s) - Late",
        (1,0): "Q-learn V(s) - Early", (1,1): "Q-learn V(s) - Middle", (1,2): "Q-learn V(s) - Late"
    }

    v_data_map = {
        (0,0): dqn_v_grids_for_plot['early'], (0,1): dqn_v_grids_for_plot['middle'], (0,2): dqn_v_grids_for_plot['late'],
        (1,0): ql_v_tables_for_plot['early'], (1,1): ql_v_tables_for_plot['middle'], (1,2): ql_v_tables_for_plot['late']
    }

    legend_handles_dict = {}
    stage_keys = ['early', 'middle', 'late'] # To map column index to stage key

    for r_ax_idx in range(2):
        for c_ax_idx in range(3):
            ax = axes[r_ax_idx, c_ax_idx]
            v_data_current = v_data_map[(r_ax_idx, c_ax_idx)]
            current_plot_title = plot_titles_map_v[(r_ax_idx, c_ax_idx)]
            current_stage_key = stage_keys[c_ax_idx]

            is_first_col_flag = (c_ax_idx == 0)
            is_last_row_flag = (r_ax_idx == 1)
            is_dqn_plot_flag = (r_ax_idx == 0) 

            current_mask_for_plot = None
            if not is_dqn_plot_flag: # This is a Q-learning plot (r_ax_idx == 1)
                current_mask_for_plot = ql_unvisited_masks[current_stage_key]
            
            _, plot_specific_handles = visualize_value_function_heatmap(
                v_data_current, ca_edges_vis, temp_edges_vis, goal_ca_plot_val,
                ax, title=current_plot_title,
                is_first_col=is_first_col_flag, is_last_row=is_last_row_flag,
                is_dqn_plot=is_dqn_plot_flag,
                unvisited_mask=current_mask_for_plot # Pass the mask
            )

            if initial_cont_state_for_plot is not None:
                plot_y_ca = np.clip(initial_cont_state_for_plot[0], ca_edges_vis[0], ca_edges_vis[-1])
                plot_x_temp = np.clip(initial_cont_state_for_plot[1], temp_edges_vis[0], temp_edges_vis[-1])
                start_m, = ax.plot(plot_x_temp, plot_y_ca, 'p', markersize=7, markeredgecolor='k',
                                   markerfacecolor='yellow', label='Start State', zorder=15, clip_on=False)
                if 'Start State' not in legend_handles_dict: legend_handles_dict['Start State'] = start_m

            if plot_specific_handles:
                for handle in plot_specific_handles:
                    label = handle.get_label()
                    if label and label not in legend_handles_dict : legend_handles_dict[label] = handle

    if legend_handles_dict:
        fig.legend(handles=list(legend_handles_dict.values()),
                   labels=list(legend_handles_dict.keys()),
                   loc='lower center', ncol=len(legend_handles_dict),
                   bbox_to_anchor=(0.5, 0.02), fontsize=9)

    fig.suptitle("State-Value Function V(s) Evolution: DQN vs. Q-learning", fontsize=14, y=0.97)
    plt.savefig('src/figs/DQN_Q_learning_plot.svg')
    plt.show()

    if continuous_env_for_dqn: continuous_env_for_dqn.close()
    if q_learning_env_wrapped: q_learning_env_wrapped.close()
    print("Script finished.")