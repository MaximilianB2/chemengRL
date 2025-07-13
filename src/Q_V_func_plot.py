import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import time

import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon, Wedge, Rectangle
from matplotlib.lines import Line2D

# Professional LaTeX setup (matching the DQN comparison style)
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 14,
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}'
})

# Professional color scheme (matching the DQN comparison)
COLORS = {
    'primary': '#2E3440',      # Dark blue-gray
    'secondary': '#5E81AC',    # Blue
    'accent': '#BF616A',       # Red
    'goal': '#D08770',         # Orange
    'start': '#aec7e8',        # Light blue for start marker
    'grid': '#4C566A',         # Gray
    'background': '#ECEFF4',   # Light gray
    'dqn': '#81A1C1',          # Light blue for DQN
    'qlearn': '#A3BE8C',       # Green for Q-learning
    'unvisited': '#E5E9F0'     # Very light gray for unvisited
}

try:
    from env.cstr import DiscreteReactorWrapper
    from pcgym import make_env
    _REAL_ENV_AVAILABLE = True
except ImportError as e:
    print(f"Error importing environment: {e}")
    print("Please ensure 'env' and 'pcgym' directories are in the Python path or current directory.")
    print("Creating dummy environment classes for demonstration purposes.")
    _REAL_ENV_AVAILABLE = False
    
    class DummyEnv:
        def __init__(self, n_actions=7):
            self._n_actions = n_actions
            self.observation_space = type('space', (object,), {'shape': (2,)})()
            self.action_space = type('space', (object,), {'n': self._n_actions})()
            self.state = np.array([0,0])
            self._max_episode_steps = 50

        def reset(self, seed=None):
            if seed is not None: np.random.seed(seed)
            self.state = np.array([np.random.randint(0,10), np.random.randint(0,10)])
            return self.state, {}

        def step(self, action):
            self.state[0] = np.clip(self.state[0] + np.random.randint(-1, 2), 0, 9)
            self.state[1] = np.clip(self.state[1] + np.random.randint(-1, 2), 0, 9)
            reward = -np.sqrt((self.state[0] - 8)**2 + (self.state[1] - 5)**2)
            terminated = False
            truncated = False
            return self.state, reward, terminated, truncated, {}

        def close(self):
            pass

    class DiscreteReactorWrapper:
        def __init__(self, env, grid_size, goal_state):
            self.env = env
            self.grid_size = grid_size
            self.n_actions = env.action_space.n
            self.goal_state = goal_state
            self.ca_bins = np.linspace(0.7, 0.9, self.grid_size[0] + 1)
            self.temp_bins = np.linspace(315., 335., self.grid_size[1] + 1)
            self.x0_continuous = np.array([0.8, 325., 0.85])
            print(f"WARNING: Using dummy DiscreteReactorWrapper with {self.n_actions} actions.")

        def reset(self, seed=None):
             state_discrete, info = self.env.reset(seed=seed)
             return state_discrete, info

        def step(self, action):
             next_state_discrete, reward, terminated, truncated, info = self.env.step(action)
             return next_state_discrete, reward*1000, terminated, truncated, info

        def close(self):
            self.env.close()

    def make_env(params):
        print("WARNING: Using dummy make_env returning DummyEnv.")
        dummy_actions = params.get('dummy_n_actions', 7)
        return DummyEnv(n_actions=dummy_actions)

def create_reactor_env(use_dummy=False, n_dummy_actions=7):
    if use_dummy or not _REAL_ENV_AVAILABLE:
        if not use_dummy:
             print("Falling back to dummy environment.")
        print(f"Creating dummy environment setup with {n_dummy_actions} actions...")
        grid_s = (10, 10)
        goal_st = 0.86
        dummy_params = {'dummy_n_actions': n_dummy_actions}
        dummy_base_env = make_env(dummy_params)
        wrapped_env = DiscreteReactorWrapper(dummy_base_env, grid_size=grid_s, goal_state=goal_st)
        print(f"Dummy wrapper configured with n_actions = {wrapped_env.n_actions}")
        return wrapped_env

    try:
        print("Attempting to create real environment...")
        nsteps = 25
        T = 26.0
        goal_state_val = 0.86
        SP = {'Ca': [goal_state_val for i in range(int(nsteps))]}
        action_space = {'low': np.array([295.]), 'high': np.array([302.])}
        observation_space = {
            'low': np.array([0.7, 315., 0.8], dtype=np.float32),
            'high': np.array([0.9, 335., 0.9], dtype=np.float32),
        }
        r_scale = {'Ca': 1e3}
        initial_x0_continuous = np.array([0.71, 326., 0.85])
        env_params = {
            'N': nsteps, 'tsim': T, 'SP': SP, 'o_space': observation_space,
            'a_space': action_space, 'x0': initial_x0_continuous.copy(),
            'model': 'cstr', 'r_scale': r_scale, 'normalise_a': True,
            'normalise_o': True, 'noise': True, 'integration_method': 'casadi',
            'noise_percentage': 0,
        }
        env = make_env(env_params)
        env.x0_continuous_unnormalized = initial_x0_continuous.copy()

        grid_size = (10, 10)
        wrapped_env = DiscreteReactorWrapper(env, grid_size=grid_size, goal_state=goal_state_val)

        if not hasattr(wrapped_env, 'n_actions'):
             print(f"WARNING: Real DiscreteReactorWrapper does not define n_actions. Assuming {n_dummy_actions}.")
             wrapped_env.n_actions = n_dummy_actions
        else:
             print(f"Real DiscreteReactorWrapper reports n_actions = {wrapped_env.n_actions}")
        
        wrapped_env.x0_continuous = initial_x0_continuous.copy()
        print(f"Using real environment. n_actions = {wrapped_env.n_actions}")
        return wrapped_env

    except Exception as e:
        print(f"Failed to create real environment: {e}")
        print("Falling back to dummy environment due to real env creation failure.")
        return create_reactor_env(use_dummy=True, n_dummy_actions=n_dummy_actions)

class QLearningAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.1, discount_factor=0.99,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration_rate
        self.q_table = np.zeros(state_dim + (action_dim,))
        print(f"Initialized Q-table with shape: {self.q_table.shape}")
        self.rewards_history = []
        self.epsilon_history = []

    def choose_action(self, state):
        state_tuple = tuple(state)
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
        else:
            q_values = self.q_table[state_tuple]
            best_actions = np.flatnonzero(q_values == np.max(q_values))
            action = np.random.choice(best_actions)
        return action

    def update(self, state, action, reward, next_state, done):
        reward *= 1000  # Scale reward for better learning
        state_tuple = tuple(state)
        next_state_tuple = tuple(next_state)
        current_q = self.q_table[state_tuple][action]
        best_next_q = np.max(self.q_table[next_state_tuple]) if not done else 0.0
        target_q = reward + self.gamma * best_next_q
        td_error = target_q - current_q
        self.q_table[state_tuple][action] = current_q + self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)

def visualize_value_function_professional(v_func_ca_temp, ca_edges, temp_edges, goal_ca_val, ax, 
                                        unvisited_mask=None, vmin=None, vmax=None):
    """Professional value function visualization matching DQN comparison style"""
    
    # Use the same pastel colormap as the DQN comparison
    cmap_val = mcolors.LinearSegmentedColormap.from_list(
        "custom_pastel", ["#54a3e3ff", "#c584da"]
    )
    
    plot_extent = [temp_edges[0], temp_edges[-1], ca_edges[0], ca_edges[-1]]
    
    # Handle unvisited states
    value_display_data = v_func_ca_temp
    if unvisited_mask is not None and np.any(unvisited_mask):
        value_display_data = np.ma.array(v_func_ca_temp, mask=unvisited_mask)
        cmap_val.set_bad(color=COLORS['unvisited'])
    
    # Always normalize values using provided vmin/vmax or computed from data
    if isinstance(value_display_data, np.ma.MaskedArray):
        compressed_data = value_display_data.compressed()
        v_finite = compressed_data[np.isfinite(compressed_data)]
    else:
        v_finite = value_display_data[np.isfinite(value_display_data)]

    if vmin is not None and vmax is not None:
        v_min, v_max = vmin, vmax
    elif v_finite.size > 0:
        v_min, v_max = np.min(v_finite), np.max(v_finite)
    else:
        v_min, v_max = -1.0, 1.0

    if abs(v_min - v_max) < 1e-9:
        v_min -= 0.5
        v_max += 0.5

    norm = mcolors.Normalize(vmin=v_min, vmax=v_max)

    # Create the heatmap
    im = ax.imshow(value_display_data, extent=plot_extent, aspect='auto', cmap=cmap_val, 
                   origin='lower', interpolation='nearest', norm=norm, zorder=1, alpha=1.0)

    # Goal line with professional styling
    goal_line = ax.axhline(y=goal_ca_val, color='#98df8a', linestyle='--', linewidth=2, zorder=10)
    
    # Set limits and ticks
    ax.set_xlim(temp_edges[0], temp_edges[-1])
    ax.set_ylim(ca_edges[0], ca_edges[-1])
    
    # Reduce number of ticks for compactness
    num_x_ticks = 4
    num_y_ticks = 4
    ax.set_xticks(np.linspace(temp_edges[0], temp_edges[-1], num_x_ticks))
    ax.set_yticks(np.linspace(ca_edges[0], ca_edges[-1], num_y_ticks))
    
    ax.set_xticklabels([f"{t:.0f}" for t in ax.get_xticks()], rotation=0, ha='center', fontsize=14)
    ax.set_yticklabels([f"{ca:.2f}" for ca in ax.get_yticks()], fontsize=14)
    
    # Professional grid
    # ax.grid(True, which='major', color=COLORS['grid'], linestyle='-', linewidth=0.2, alpha=0.3, zorder=0)
    
    return im, goal_line

def visualize_q_function_professional(q_func_full, ca_edges_q, temp_edges_q, goal_ca_q, n_actions_q, ax,
                                    unvisited_mask=None, vmin=None, vmax=None):
    """Professional Q-function visualization with action strips"""
    
    num_ca_cells, num_temp_cells, num_actions_from_q = q_func_full.shape
    if n_actions_q != num_actions_from_q:
        print(f"Warning: n_actions provided ({n_actions_q}) differs from q_func_full.shape[2] ({num_actions_from_q}). Using {num_actions_from_q}.")
        n_actions_q = num_actions_from_q
    if n_actions_q <= 0: 
        return None, []

    # Use the same pastel colormap
    cmap_qval = mcolors.LinearSegmentedColormap.from_list(
        "custom_pastel", ["#54a3e3ff", "#c584da"]
    )
    
    # Normalization
    if vmin is not None and vmax is not None:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        q_finite = q_func_full[np.isfinite(q_func_full)]
        if q_finite.size == 0: 
            q_min, q_max = -1.0, 1.0
        else:
            q_min, q_max = np.min(q_finite), np.max(q_finite)
            if abs(q_min - q_max) < 1e-9: 
                q_min -= 0.5; q_max += 0.5
        norm = mcolors.Normalize(vmin=q_min, vmax=q_max)
    
    default_color = COLORS['unvisited']

    for r in range(num_ca_cells):
        ca_low, ca_high = ca_edges_q[r], ca_edges_q[r+1]
        cell_height = ca_high - ca_low
        for c in range(num_temp_cells):
            temp_low, temp_high = temp_edges_q[c], temp_edges_q[c+1]
            cell_width = temp_high - temp_low
            
            q_s_current = q_func_full[r, c, :]

            # Check if this state is unvisited
            if unvisited_mask is not None and unvisited_mask[r, c]:
                grey_cell_rect = Rectangle((temp_low, ca_low), cell_width, cell_height,
                                         facecolor=default_color, edgecolor='silver', linewidth=0.2, zorder=0)
                ax.add_patch(grey_cell_rect)
                continue

            # Find best action for highlighting
            best_action_to_highlight = -1
            finite_q_s_vals = q_s_current[np.isfinite(q_s_current)]
            if finite_q_s_vals.size > 0:
                max_q_for_state = np.max(finite_q_s_vals)
                candidate_indices = np.where(q_s_current == max_q_for_state)[0]
                if candidate_indices.size > 0:
                    best_action_to_highlight = candidate_indices[0]

            strip_width = cell_width / n_actions_q
            for action_idx in range(n_actions_q):
                q_val = q_s_current[action_idx]
                color = cmap_qval(norm(q_val)) if np.isfinite(q_val) else default_color
                
                edge_color_for_strip = COLORS['grid']
                line_width_for_strip = 0.2
                z_order_for_strip = 1

                if action_idx == best_action_to_highlight and np.isfinite(q_val):
                    edge_color_for_strip = COLORS['primary']
                    line_width_for_strip = 2.0
                    z_order_for_strip = 5

                strip_x_low = temp_low + action_idx * strip_width
                rect = Rectangle((strip_x_low, ca_low), strip_width, cell_height,
                               facecolor=color, edgecolor=edge_color_for_strip,
                               linewidth=line_width_for_strip, zorder=z_order_for_strip)
                ax.add_patch(rect)

    # Goal line with professional styling
    goal_line = ax.axhline(y=goal_ca_q, color='#98df8a', linestyle='--', linewidth=2, 
                          label=f'Goal', zorder=10)
    
    # Set limits and ticks
    ax.set_xlim(temp_edges_q[0], temp_edges_q[-1])
    ax.set_ylim(ca_edges_q[0], ca_edges_q[-1])
    
    num_x_ticks = 4
    num_y_ticks = 4
    ax.set_xticks(np.linspace(temp_edges_q[0], temp_edges_q[-1], num_x_ticks))
    ax.set_yticks(np.linspace(ca_edges_q[0], ca_edges_q[-1], num_y_ticks))
    
    ax.set_xticklabels([f"{t:.0f}" for t in ax.get_xticks()], rotation=0, ha='center', fontsize=14)
    ax.set_yticklabels([f"{ca:.2f}" for ca in ax.get_yticks()], fontsize=14)
    
    # Professional grid
    # ax.grid(True, which='major', color=COLORS['grid'], linestyle='-', linewidth=0.2, alpha=0.3, zorder=0)
    
    return norm, [goal_line]

def add_action_temperature_key_professional(ax, num_actions_key):
    """Professional action key matching the DQN comparison style"""
    if num_actions_key <= 0: 
        return
    
    # Create a color gradient for temperature effects
    colors = ['#5E81AC', '#ECEFF4', '#BF616A']  # Blue -> Light -> Red
    cmap_legend = mcolors.LinearSegmentedColormap.from_list("temp_effect", colors)
    
    # Position the key
    ex_x, ex_y, ex_w, ex_h = 0.25, 1.08, 0.50, 0.04
    strip_w = ex_w / num_actions_key
    
    # # Add border
    # ax.add_patch(Rectangle((ex_x, ex_y), ex_w, ex_h, fill=False, edgecolor=COLORS['primary'], 
    #                       lw=1, clip_on=False, transform=ax.transAxes))
    
    # Add title
    # ax.text(ex_x + ex_w / 2, ex_y + ex_h + 0.015, 'Action Effect on Temperature', 
    #        ha='center', va='bottom', transform=ax.transAxes, fontsize=12, weight='bold')
    
    # # Add color strips
    # for i in range(num_actions_key):
    #     strip_x = ex_x + i * strip_w
    #     color_val = i / (num_actions_key - 1) if num_actions_key > 1 else 0.5
    #     ax.add_patch(Rectangle((strip_x, ex_y), strip_w, ex_h, 
    #                           facecolor=cmap_legend(color_val), edgecolor=COLORS['grid'], 
    #                           lw=0.3, clip_on=False, transform=ax.transAxes))
    
    # Add labels
    # ax.text(ex_x, ex_y - 0.015, 'Decrease', ha='center', va='top', 
    #        transform=ax.transAxes, fontsize=10)
    # ax.text(ex_x + ex_w, ex_y - 0.015, 'Increase', ha='center', va='top', 
    #        transform=ax.transAxes, fontsize=10)

# --- Create environment and train agent ---
N_ACTIONS_TO_USE = 6
disc_env = create_reactor_env(use_dummy=False, n_dummy_actions=N_ACTIONS_TO_USE)

agent = QLearningAgent(
    state_dim=disc_env.grid_size,
    action_dim=disc_env.n_actions,
    learning_rate=0.1,
    discount_factor=0.99,
    exploration_rate=1.0,
    exploration_decay=0.99,
    min_exploration_rate=0.01
)

# Training Loop
num_episodes = 300
max_steps_per_episode = 25

progress_bar = tqdm(range(num_episodes), desc="Training Progress")
episode_rewards = []

for episode in progress_bar:
    state, _ = disc_env.reset()
    total_reward = 0
    terminated, truncated = False, False
    step = 0
    while not (terminated or truncated) and step < max_steps_per_episode:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = disc_env.step(action)
        is_done_for_q_update = terminated or truncated
        agent.update(state, action, reward, next_state, is_done_for_q_update)
        state = next_state
        total_reward += reward
        step += 1
    agent.decay_epsilon()
    episode_rewards.append(total_reward)
    progress_bar.set_postfix({'Last Reward': f"{total_reward:.2f}", 'Epsilon': f"{agent.epsilon:.3f}"})

disc_env.close()

# --- Prepare data for visualization ---
value_function = np.max(agent.q_table, axis=2)
q_function = agent.q_table
unvisited_mask = np.all(agent.q_table == 0, axis=2)

# Get environment parameters
ca_edges = disc_env.ca_bins
temp_edges = disc_env.temp_bins
goal_state_plot_val = disc_env.goal_state
n_actions_val = disc_env.n_actions

# Get initial state for plotting
initial_continuous_state_for_plot = None
if hasattr(disc_env, 'x0_continuous'):
    initial_continuous_state_for_plot = disc_env.x0_continuous
elif hasattr(disc_env, 'env') and hasattr(disc_env.env, 'x0_continuous_unnormalized'):
    initial_continuous_state_for_plot = disc_env.env.x0_continuous_unnormalized

# Calculate shared normalization for both plots
all_finite_v = value_function[np.isfinite(value_function) & ~unvisited_mask]
all_finite_q = q_function[np.isfinite(q_function) & ~np.repeat(unvisited_mask[:,:,np.newaxis], q_function.shape[2], axis=2)]

if len(all_finite_v) > 0 and len(all_finite_q) > 0:
    vmin_shared = min(np.min(all_finite_v), np.min(all_finite_q))
    vmax_shared = max(np.max(all_finite_v), np.max(all_finite_q))
    if abs(vmin_shared - vmax_shared) < 1e-9:
        vmin_shared -= 0.5; vmax_shared += 0.5
else:
    vmin_shared, vmax_shared = -1.0, 1.0

# --- Create Professional Visualization ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(hspace=0.25, wspace=0.15, top=0.85, bottom=0.15, left=0.08, right=0.8)

# Add main title
# fig.suptitle('Q-Learning: Value and Action-Value Functions', fontsize=16, fontweight='bold', y=0.95)

# Value function plot
im_v, goal_line_v = visualize_value_function_professional(
    value_function, ca_edges, temp_edges, goal_state_plot_val, axes[0],
    unvisited_mask=unvisited_mask, vmin=vmin_shared, vmax=vmax_shared
)
axes[0].set_title(r'State Value Function', fontsize=14)
axes[0].set_ylabel(r'$C_A$ (mol/L)', fontsize=14)
axes[0].set_xlabel(r'$T$ (K)', fontsize=14)

# Q-function plot
norm_q, goal_line_q = visualize_q_function_professional(
    q_function, ca_edges, temp_edges, goal_state_plot_val, n_actions_val, axes[1],
    unvisited_mask=unvisited_mask, vmin=vmin_shared, vmax=vmax_shared
)
axes[1].set_title(r'Action-Value Function', fontsize=14)
axes[1].set_xlabel(r'$T$ (K)', fontsize=14)
axes[1].set_yticklabels([])

# Add action temperature key to Q-function plot
add_action_temperature_key_professional(axes[1], n_actions_val)

# Add start state markers
legend_handles = []
if initial_continuous_state_for_plot is not None:
    init_ca_val_plot = np.clip(initial_continuous_state_for_plot[0], ca_edges[0], ca_edges[-1])
    init_temp_val_plot = np.clip(initial_continuous_state_for_plot[1], temp_edges[0], temp_edges[-1])
    
    start_marker_v, = axes[0].plot(init_temp_val_plot, init_ca_val_plot, 'o', markersize=7.5, 
                                  markeredgecolor=COLORS['primary'], markerfacecolor=COLORS['start'], 
                                  zorder=15, clip_on=False)
    start_marker_q, = axes[1].plot(init_temp_val_plot, init_ca_val_plot, 'o', markersize=7.5, 
                                  markeredgecolor=COLORS['primary'], markerfacecolor=COLORS['start'], 
                                  zorder=15, clip_on=False)
    
    legend_handles.extend([goal_line_v, start_marker_v])

# Add shared colorbar
if im_v is not None:
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im_v, cax=cbar_ax)
    cbar.set_label(r'$V(\textbf{x})$ or Q(\textbf{x},\textbf{u})', fontsize=14)

# Add legend
if legend_handles:
    fig.legend(handles=legend_handles, labels=['Goal', r'$\textbf{x}_0$'], 
              loc='upper center', ncol=2, bbox_to_anchor=(0.45, 0.08), 
              fontsize=14, frameon=False)

plt.savefig('Q_V_function_plot.pdf', bbox_inches='tight', dpi=300)
plt.show()

# Learning curve with professional styling
fig, ax = plt.subplots(figsize=(10, 4), tight_layout=True)
ax.plot(episode_rewards, alpha=0.7, color=COLORS['qlearn'], linewidth=0.8, label='Episode Reward')

# Moving average
window_size = min(50, len(episode_rewards) // 5)
if len(episode_rewards) >= window_size and window_size > 0:
    moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    ax.plot(np.arange(window_size - 1, len(episode_rewards)), moving_avg, 
            color=COLORS['accent'], linewidth=1.5, label=f'{window_size}-Episode Average')

ax.set_xlabel('Episode', fontsize=14)
ax.set_ylabel('Total Reward', fontsize=14)
ax.set_title('Q-Learning Training Progress', fontsize=14)
ax.legend(frameon=True, fancybox=False, shadow=False, framealpha=0.9, edgecolor=COLORS['grid'])
# ax.grid(True, alpha=0.3, color=COLORS['grid'], linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('professional_learning_curve.pdf', bbox_inches='tight', dpi=300)
plt.show()