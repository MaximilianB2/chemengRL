import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from collections import deque
import copy

import torch
import torch.nn as nn
import torch.optim as optim

# Professional LaTeX setup
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 12,
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}'
})

# Professional color scheme
COLORS = {
    'primary': '#2E3440',      # Dark blue-gray
    'secondary': '#5E81AC',    # Blue
    'accent': '#98df8a',       # Red
    'goal': '#D08770',         # Orange
    'start': '#aec7e8',        # Yellow
    'end': '#A3BE8C',          # Green
    'grid': '#4C566A',         # Gray
    'background': '#ECEFF4',   # Light gray
    'dataset': '#7f7f7f',      # Light blue
    'rollout': '#c5b0d5'       # Purple
}

_PCGYM_AVAILABLE = False
try:
    from pcgym import make_env
    _PCGYM_AVAILABLE = True
    print("pcgym imported successfully.")
except ImportError as e:
    print(f"Error importing pcgym: {e}")
    print("DQN training, dataset generation, CQL training, and plotting will be skipped.")

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim1=128, hidden_dim2=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQN_Agent:
    def __init__(self, state_size, num_discrete_actions, continuous_action_bounds, device="cpu"):
        self.state_size = state_size
        self.num_discrete_actions = num_discrete_actions
        self.memory = deque(maxlen=20000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.device = device
        self.update_target_freq = 10
        self.train_step_counter = 0

        self.action_low_cont = continuous_action_bounds['low'][0]
        self.action_high_cont = continuous_action_bounds['high'][0]
        if self.num_discrete_actions > 1:
            self.discrete_actions_map_to_continuous = np.linspace(self.action_low_cont, self.action_high_cont, self.num_discrete_actions)
        elif self.num_discrete_actions == 1:
             self.discrete_actions_map_to_continuous = np.array([(self.action_low_cont + self.action_high_cont)/2])
        else:
            self.discrete_actions_map_to_continuous = np.array([])

        self.q_network = QNetwork(state_size, num_discrete_actions).to(device)
        self.target_network = QNetwork(state_size, num_discrete_actions).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action_idx, reward, next_state, done):
        self.memory.append((state, action_idx, reward, next_state, done))

    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.num_discrete_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state_tensor)
            self.q_network.train()
            action_idx = torch.argmax(action_values).item()
        
        continuous_action_val = np.array([self.discrete_actions_map_to_continuous[action_idx]])
        return continuous_action_val, action_idx

    def train_batch(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([exp[0] for exp in minibatch])
        actions_idx = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
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
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.train_step_counter += 1
        if self.train_step_counter % self.update_target_freq == 0:
            self.update_target_network()
        return loss.item()

class CQL_Agent(DQN_Agent):
    def __init__(self, state_size, num_discrete_actions, continuous_action_bounds,
                 device="cpu", cql_alpha=5.0, learning_rate=0.0005,
                 gamma=0.99, batch_size=64, target_update_freq=10, memory_size=20000):
        super().__init__(state_size, num_discrete_actions, continuous_action_bounds, device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.cql_alpha = cql_alpha
        self.learning_rate = learning_rate 
        self.update_target_freq = target_update_freq
        self.train_step_counter = 0
        self.memory = deque(maxlen=memory_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.epsilon = 0.0 
        self.epsilon_min = 0.0
        self.epsilon_decay = 1.0

    def train_batch(self):
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0, 0.0

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([exp[0] for exp in minibatch])
        actions_idx = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_idx_tensor = torch.LongTensor(actions_idx).unsqueeze(1).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        current_q_values_for_dataset_actions = self.q_network(states_tensor).gather(1, actions_idx_tensor).squeeze(1)
        with torch.no_grad():
            next_actions_indices_online_net = self.q_network(next_states_tensor).argmax(dim=1, keepdim=True)
            next_q_values_target_net = self.target_network(next_states_tensor).gather(1, next_actions_indices_online_net).squeeze(1)
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values_target_net
        bellman_loss = nn.MSELoss()(current_q_values_for_dataset_actions, target_q_values)

        q_values_all_actions_current_net = self.q_network(states_tensor)
        log_sum_exp_q = torch.logsumexp(q_values_all_actions_current_net, dim=1)
        cql_diff = log_sum_exp_q - current_q_values_for_dataset_actions
        cql_term_loss = cql_diff.mean()
        
        total_loss = bellman_loss + self.cql_alpha * cql_term_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.train_step_counter += 1
        if self.train_step_counter % self.update_target_freq == 0:
            self.update_target_network()
        
        return total_loss.item(), cql_term_loss.item(), bellman_loss.item()

    def act(self, state, training=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state_tensor)
        self.q_network.train()
        action_idx = torch.argmax(action_values).item()
        
        if not self.discrete_actions_map_to_continuous.any():
            continuous_action_val = np.array([0.0]) 
        else:
            continuous_action_val = np.array([self.discrete_actions_map_to_continuous[action_idx]])
        return continuous_action_val, action_idx

def create_continuous_reactor_env_for_dqn():
    if not _PCGYM_AVAILABLE:
        return None
    try:
        nsteps = 60
        T = 26.0
        goal_ca_cont = 0.86
        SP = {'Ca': [goal_ca_cont for _ in range(int(nsteps))]}

        action_space_cont = {'low': np.array([295.]), 'high': np.array([302.])}
        
        observation_space_cont = {
            'low': np.array([0.7, 315., 0.8], dtype=np.float32),
            'high': np.array([0.9, 335., 0.9], dtype=np.float32),
        }
        r_scale = {'Ca': 1e3}
        initial_x0 = np.array([0.72, 328., goal_ca_cont])

        env_params = {
            'N': nsteps, 'tsim': T, 'SP': SP, 
            'o_space': observation_space_cont, 'a_space': action_space_cont, 
            'x0': initial_x0, 'model': 'cstr', 'r_scale': r_scale,
            'normalise_a': True, 'normalise_o': True, 'noise': True, 
            'integration_method': 'casadi', 'noise_percentage': 0.001
        }
        cont_env = make_env(env_params)
        cont_env.goal_ca_continuous = goal_ca_cont
        print("Continuous CSTR environment for DQN/CQL created successfully.")
        return cont_env
    except Exception as e:
        print(f"Error creating continuous CSTR environment for DQN/CQL: {e}")
        return None

def train_dqn_agent(dqn_env, dqn_agent, episodes, dqn_action_size):
    if dqn_env is None or dqn_agent is None:
        return [], {}
    
    print(f"Starting DQN training for {episodes} episodes...")
    dqn_scores = []
    dqn_q_func_snapshots = {'early': None, 'middle': None, 'late': None}
    ep_early = max(1, int(episodes * 0.05))
    ep_middle = max(ep_early + 1, int(episodes * 0.5))
    
    snapshot_agent = DQN_Agent(dqn_agent.state_size, dqn_action_size,
                               {'low': np.array([dqn_agent.action_low_cont]), 
                                'high': np.array([dqn_agent.action_high_cont])},
                               device=dqn_agent.device)
    
    progress_bar = tqdm(range(episodes), desc="DQN Training")
    for e in progress_bar:
        state_cont, _ = dqn_env.reset()
        total_reward_ep = 0
        for step_num in range(dqn_env.N):
            continuous_action_val, action_idx = dqn_agent.act(state_cont, training=True)
            next_state_cont, reward, terminated, truncated, _ = dqn_env.step(continuous_action_val)
            done = terminated or truncated
            
            dqn_agent.remember(state_cont, action_idx, reward, next_state_cont, done)
            state_cont = next_state_cont
            total_reward_ep += reward
            loss = dqn_agent.train_batch()
            if done:
                break
        dqn_scores.append(total_reward_ep)
        progress_bar.set_postfix({'R': f"{total_reward_ep:.1f}", 
                                  'Eps': f"{dqn_agent.epsilon:.2f}", 
                                  'L': f"{loss:.3f}"})
        
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

def plot_reward_curve(scores, algo_name, color):
    fig, ax = plt.subplots(figsize=(8, 3), tight_layout=True)
    ax.plot(scores, alpha=0.7, color=color, linewidth=0.8, label=f'{algo_name}')
    
    window = max(1, len(scores) // 20)
    if len(scores) >= window and window > 0:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window - 1, len(scores)), moving_avg, 
                color=COLORS['accent'], linewidth=1.5, label=f'{window}-Episode Average')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend(frameon=True, fancybox=False, shadow=False, framealpha=0.9, edgecolor=COLORS['grid'])
    # ax.grid(True, alpha=0.3, color=COLORS['grid'], linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

OBS_LOW_ORIG_FOR_DQN_ENV = np.array([0.7, 315., 0.8], dtype=np.float32)
OBS_HIGH_ORIG_FOR_DQN_ENV = np.array([0.9, 335., 0.9], dtype=np.float32)

def denormalize_state_vector(norm_state_vec, low_orig, high_orig):
    norm_state_vec = np.asarray(norm_state_vec)
    low_orig = np.asarray(low_orig)
    high_orig = np.asarray(high_orig)
    return (norm_state_vec + 1.0) / 2.0 * (high_orig - low_orig) + low_orig

def plot_dataset_and_cql_rollout(dataset_trajectories_normalized_states,
                                 cql_rollout_normalized_states,
                                 ca_plot_lims_orig_edges, temp_plot_lims_orig_edges,
                                 goal_ca_val_orig):
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)

    # Plot dataset trajectories
    if dataset_trajectories_normalized_states:
        for i, trajectory_norm in enumerate(dataset_trajectories_normalized_states):
            if not trajectory_norm: continue
            traj_points_norm = np.array(trajectory_norm)
            traj_orig = denormalize_state_vector(traj_points_norm[:, :2], 
                                                 OBS_LOW_ORIG_FOR_DQN_ENV[:2], 
                                                 OBS_HIGH_ORIG_FOR_DQN_ENV[:2])
            
            label = 'Dataset' if i == 0 else None
            ax.plot(traj_orig[:, 1], traj_orig[:, 0], linewidth=1.0, alpha=0.4, 
                    color=COLORS['dataset'], label=label, zorder=2)

    # Plot CQL agent rollout trajectory
    if cql_rollout_normalized_states:
        rollout_np_norm = np.array(cql_rollout_normalized_states)
        rollout_orig = denormalize_state_vector(rollout_np_norm[:, :2], 
                                                OBS_LOW_ORIG_FOR_DQN_ENV[:2], 
                                                OBS_HIGH_ORIG_FOR_DQN_ENV[:2])
        ax.plot(rollout_orig[:, 1], rollout_orig[:, 0], color=COLORS['rollout'], 
                linewidth=1.5, label="CQL", zorder=3)
        if len(rollout_orig) > 0:
            ax.plot(rollout_orig[0, 1], rollout_orig[0, 0], 'o', markersize=8, 
                    color=COLORS['start'], markeredgecolor=COLORS['primary'], 
                    label=r"$\textbf{x}_0$", zorder=4)
    

    ax.axhline(y=goal_ca_val_orig, color='#98df8a', linestyle='--', 
               linewidth=2, label=r'Goal', zorder=1)

    ax.set_xlabel(r"Temperature $T$ (K)")
    ax.set_ylabel(r"Concentration $C_A$ (mol/L)")
    
    ax.legend(frameon=False, fancybox=False, shadow=False, framealpha=0.9, edgecolor=COLORS['grid'],
              ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.13), columnspacing=0.8)
    # ax.grid(True, alpha=0.3, color=COLORS['grid'], linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlim(temp_plot_lims_orig_edges[0], temp_plot_lims_orig_edges[-1])
    ax.set_ylim(ca_plot_lims_orig_edges[0], ca_plot_lims_orig_edges[-1])

    plt.savefig('offline_plot.pdf', bbox_inches='tight', dpi=300)
    plt.show()

# Main Script
if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    VIS_GRID_SIZE = (10, 10)
    GOAL_CA_SHARED = 0.86

    DQN_EPISODES_FOR_POLICY = 40 if _PCGYM_AVAILABLE else 0
    DQN_ACTION_SIZE = 21
    DATA_COLLECTION_EPISODES_DQN = 10
    CQL_TRAINING_STEPS = 1000
    CQL_BATCH_SIZE = 64
    CQL_ALPHA = 5.0
    CQL_LEARNING_RATE = 1e-4
    CQL_TARGET_UPDATE_FREQ = 10

    continuous_env_for_dqn = None
    dqn_agent = None
    dqn_state_size = 3
    dqn_action_bounds = {'low': np.array([295.]), 'high': np.array([302.])} 
    
    dqn_scores_history = []
    dqn_q_func_state_dicts = {'early': None, 'middle': None, 'late': None}

    if _PCGYM_AVAILABLE and DQN_EPISODES_FOR_POLICY > 0:
        continuous_env_for_dqn = create_continuous_reactor_env_for_dqn()
        if continuous_env_for_dqn:
            dqn_state_size = continuous_env_for_dqn.observation_space.shape[0]
            dqn_action_bounds = {'low': continuous_env_for_dqn.action_space.low, 
                                 'high': continuous_env_for_dqn.action_space.high}
            
            dqn_agent_for_policy = DQN_Agent(dqn_state_size, DQN_ACTION_SIZE, dqn_action_bounds, device=DEVICE)
            dqn_scores_history, dqn_q_func_state_dicts = train_dqn_agent(
                continuous_env_for_dqn, dqn_agent_for_policy, DQN_EPISODES_FOR_POLICY, DQN_ACTION_SIZE)
            plot_reward_curve(dqn_scores_history, "DQN", COLORS['secondary'])
    else: 
        print("Skipping DQN training for data policy.")

    # Data Collection for CQL
    offline_dataset_for_cql_transitions = []
    offline_dataset_trajectories_for_plot = []

    dqn_model_weights_for_dataset = dqn_q_func_state_dicts['middle']
    if dqn_model_weights_for_dataset is None and dqn_q_func_state_dicts['late'] is not None:
        dqn_model_weights_for_dataset = dqn_q_func_state_dicts['late']

    if dqn_model_weights_for_dataset is not None and continuous_env_for_dqn is not None:
        print(f"\nGenerating dataset using trained DQN agent for {DATA_COLLECTION_EPISODES_DQN} episodes...")
        data_collection_env = continuous_env_for_dqn

        if data_collection_env:
            data_collector_agent = DQN_Agent(dqn_state_size, DQN_ACTION_SIZE, dqn_action_bounds, device=DEVICE)
            data_collector_agent.q_network.load_state_dict(dqn_model_weights_for_dataset)
            data_collector_agent.update_target_network()
            data_collector_agent.epsilon = 0.1

            for ep in tqdm(range(DATA_COLLECTION_EPISODES_DQN), desc="Dataset Generation"):
                current_episode_states_normalized = []
                state_cont_norm, _ = data_collection_env.reset()
                current_episode_states_normalized.append(state_cont_norm.copy())

                for _ in range(data_collection_env.N):
                    continuous_action_val, action_idx = data_collector_agent.act(state_cont_norm, training=True)
                    next_state_cont_norm, reward, terminated, truncated, _ = data_collection_env.step(continuous_action_val)
                    done = terminated or truncated
                    
                    offline_dataset_for_cql_transitions.append((state_cont_norm.copy(), action_idx, reward, next_state_cont_norm.copy(), done))
                    current_episode_states_normalized.append(next_state_cont_norm.copy())
                    
                    state_cont_norm = next_state_cont_norm
                    if done:
                        break
                offline_dataset_trajectories_for_plot.append(current_episode_states_normalized)
            
            print(f"Generated dataset with {len(offline_dataset_for_cql_transitions)} transitions.")
    else:
        print("Skipping dataset generation: DQN model or environment not available.")

    # CQL Agent Training
    cql_agent = None
    if offline_dataset_for_cql_transitions and continuous_env_for_dqn is not None:
        print(f"\nTraining CQL agent for {CQL_TRAINING_STEPS} steps...")
        cql_agent = CQL_Agent(
            state_size=dqn_state_size, num_discrete_actions=DQN_ACTION_SIZE,
            continuous_action_bounds=dqn_action_bounds, device=DEVICE,
            cql_alpha=CQL_ALPHA, learning_rate=CQL_LEARNING_RATE,
            batch_size=CQL_BATCH_SIZE, target_update_freq=CQL_TARGET_UPDATE_FREQ,
            memory_size=len(offline_dataset_for_cql_transitions) + 100
        )
        for experience in offline_dataset_for_cql_transitions:
            cql_agent.remember(*experience)

        cql_total_losses, cql_term_losses, cql_bellman_losses = [], [], []
        progress_bar_cql = tqdm(range(CQL_TRAINING_STEPS), desc="CQL Training")
        for step in progress_bar_cql:
            total_loss, cql_loss, bellman_loss = cql_agent.train_batch()
            cql_total_losses.append(total_loss)
            cql_term_losses.append(cql_loss)
            cql_bellman_losses.append(bellman_loss)
            if step % 100 == 0:
                progress_bar_cql.set_postfix({'Total': f"{total_loss:.3f}", 
                                              'CQL': f"{cql_loss:.3f}", 
                                              'Bellman': f"{bellman_loss:.3f}"})
        
        # Plot CQL losses
        fig, ax = plt.subplots(figsize=(8, 3), tight_layout=True)
        ax.plot(cql_total_losses, label='Total', alpha=0.8, color=COLORS['primary'], linewidth=1.0)
        ax.plot(cql_term_losses, label=fr'CQL ($\alpha={CQL_ALPHA}$)', alpha=0.8, color=COLORS['accent'], linewidth=1.0)
        ax.plot(cql_bellman_losses, label='Bellman', alpha=0.8, color=COLORS['secondary'], linewidth=1.0)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.legend(frameon=True, fancybox=False, shadow=False, framealpha=0.9, edgecolor=COLORS['grid'])
        # ax.grid(True, alpha=0.3, color=COLORS['grid'], linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.savefig('cql_losses.pdf', bbox_inches='tight', dpi=300)
        plt.show()
    else:
        print("Skipping CQL training: No dataset or environment for CQL.")

    # CQL Agent Rollout and Plotting
    if cql_agent is not None and continuous_env_for_dqn is not None:
        print("\nGenerating CQL agent rollout for plotting...")
        rollout_env = continuous_env_for_dqn
        
        if rollout_env:
            cql_rollout_normalized_states = []
            state_norm, _ = rollout_env.reset(seed=42)
            cql_rollout_normalized_states.append(state_norm.copy())
            total_rollout_reward = 0
            for _ in range(rollout_env.N):
                continuous_action_val, _ = cql_agent.act(state_norm)
                next_state_norm, reward, terminated, truncated, _ = rollout_env.step(continuous_action_val)
                
                cql_rollout_normalized_states.append(next_state_norm.copy())
                total_rollout_reward += reward
                done = terminated or truncated
                state_norm = next_state_norm
                if done:
                    break

            ca_edges_for_plot_limits = np.linspace(OBS_LOW_ORIG_FOR_DQN_ENV[0], OBS_HIGH_ORIG_FOR_DQN_ENV[0], VIS_GRID_SIZE[0] + 1)
            temp_edges_for_plot_limits = np.linspace(OBS_LOW_ORIG_FOR_DQN_ENV[1], OBS_HIGH_ORIG_FOR_DQN_ENV[1], VIS_GRID_SIZE[1] + 1)
            goal_ca_to_plot = GOAL_CA_SHARED

            plot_dataset_and_cql_rollout(
                offline_dataset_trajectories_for_plot,
                cql_rollout_normalized_states,
                ca_edges_for_plot_limits, 
                temp_edges_for_plot_limits, 
                goal_ca_to_plot
            )
        else:
            print("Failed to get/create environment for CQL rollout.")
    else:
        print("Skipping CQL rollout and plotting.")

    if continuous_env_for_dqn:
        continuous_env_for_dqn.close()

    print("Script finished.")