import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import time # For potential seeding or delays if needed

import matplotlib.patches as patches
import matplotlib.colors as mcolors
# Import Wedge for the new visualization
from matplotlib.patches import Polygon, Wedge, Rectangle # Keep Polygon for potential future use, add Wedge
# Import Line2D for creating custom legend handles
from matplotlib.lines import Line2D
# Set LaTeX font to Times New Roman
plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times New Roman')
try:
    from env.cstr import DiscreteReactorWrapper
    from pcgym import make_env
    _REAL_ENV_AVAILABLE = True
except ImportError as e:
    print(f"Error importing environment: {e}")
    print("Please ensure 'env' and 'pcgym' directories are in the Python path or current directory.")
    # As a fallback for running the rest of the script, create dummy classes
    print("Creating dummy environment classes for demonstration purposes.")
    _REAL_ENV_AVAILABLE = False
    class DummyEnv:
        def __init__(self, n_actions=7): # Allow setting n_actions for dummy
            self._n_actions = n_actions
            self.observation_space = type('space', (object,), {'shape': (2,)})() # Dummy shape
            self.action_space = type('space', (object,), {'n': self._n_actions})() # Use n_actions
            self.state = np.array([0,0])
            self._max_episode_steps = 50

        def reset(self, seed=None):
            if seed is not None: np.random.seed(seed) # Basic seeding
            self.state = np.array([np.random.randint(0,10), np.random.randint(0,10)])
            return self.state, {}

        def step(self, action):

            self.state[0] = np.clip(self.state[0] + np.random.randint(-1, 2), 0, 9)
            self.state[1] = np.clip(self.state[1] + np.random.randint(-1, 2), 0, 9)
            reward = -np.sqrt((self.state[0] - 8)**2 + (self.state[1] - 5)**2) # Reward for being near (8,5)
            terminated = False
            truncated = False # In dummy env, let max_steps handle this
            return self.state, reward, terminated, truncated, {}

        def close(self):
            pass

    class DiscreteReactorWrapper:
        def __init__(self, env, grid_size, goal_state):
            self.env = env # The dummy env
            self.grid_size = grid_size # e.g., (10, 10)
            # Get n_actions from the underlying dummy env
            self.n_actions = env.action_space.n
            self.goal_state = goal_state # e.g. 0.86 (continuous)
            # Define dummy bins based on typical ranges if real env failed
            self.ca_bins = np.linspace(0.7, 0.9, self.grid_size[0] + 1)
            self.temp_bins = np.linspace(315., 335., self.grid_size[1] + 1)
            # Store dummy initial continuous state if needed
            self.x0_continuous = np.array([0.8, 325., 0.85]) # Example [Ca, Temp, Ca_SP]
            print(f"WARNING: Using dummy DiscreteReactorWrapper with {self.n_actions} actions.")

        def reset(self, seed=None):
             state_discrete, info = self.env.reset(seed=seed)
             # Store continuous state if available from info (for dummy, not implemented here)
             # self.current_continuous_state = info.get('continuous_state', None)
             return state_discrete, info

        def step(self, action):
             next_state_discrete, reward, terminated, truncated, info = self.env.step(action)
             # self.current_continuous_state = info.get('continuous_state', None)
             return next_state_discrete, reward, terminated, truncated, info

        def close(self):
            self.env.close()

    def make_env(params):
        print("WARNING: Using dummy make_env returning DummyEnv.")
        # Pass n_actions to dummy env if needed, or define it here
        dummy_actions = params.get('dummy_n_actions', 7) # Get from params or default
        return DummyEnv(n_actions=dummy_actions)

# --- Environment Creation ---
def create_reactor_env(use_dummy=False, n_dummy_actions=7):
    if use_dummy or not _REAL_ENV_AVAILABLE:
        if not use_dummy: # If fallback triggered
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
        nsteps = 20
        T = 26.0
        goal_state_val = 0.85 # Note: Renamed from goal_state to avoid conflict in global scope
        SP = {'Ca': [goal_state_val for i in range(int(nsteps))]}
        action_space = {'low': np.array([295.]), 'high': np.array([302.])}
        observation_space = {
            'low': np.array([0.7, 315., 0.8], dtype=np.float32),
            'high': np.array([0.9, 335., 0.9], dtype=np.float32),
        }
        r_scale = {'Ca': 1e3}
        # Initial continuous state for the real environment [Ca, Temp, Ca_SP_initial]
        initial_x0_continuous = np.array([0.71, 326., 0.85]) # Example initial state
        env_params = {
            'N': nsteps, 'tsim': T, 'SP': SP, 'o_space': observation_space,
            'a_space': action_space, 'x0': initial_x0_continuous.copy(),
            'model': 'cstr', 'r_scale': r_scale, 'normalise_a': True,
            'normalise_o': True, 'noise': True, 'integration_method': 'casadi',
            'noise_percentage': 0,
        }
        env = make_env(env_params)
        env.x0_continuous_unnormalized = initial_x0_continuous.copy() # Store for plotting start marker

        grid_size = (10, 10)
        wrapped_env = DiscreteReactorWrapper(env, grid_size=grid_size, goal_state=goal_state_val)

        if not hasattr(wrapped_env, 'n_actions'):
             print(f"WARNING: Real DiscreteReactorWrapper does not define n_actions. Assuming {n_dummy_actions}.")
             wrapped_env.n_actions = n_dummy_actions
        else:
             print(f"Real DiscreteReactorWrapper reports n_actions = {wrapped_env.n_actions}")
        
        # Store the initial continuous state in the wrapper if possible (for consistent access)
        wrapped_env.x0_continuous = initial_x0_continuous.copy()

        print(f"Using real environment. n_actions = {wrapped_env.n_actions}")
        return wrapped_env

    except Exception as e:
        print(f"Failed to create real environment: {e}")
        print("Falling back to dummy environment due to real env creation failure.")
        return create_reactor_env(use_dummy=True, n_dummy_actions=n_dummy_actions)

# --- Create environment ---
N_ACTIONS_TO_USE = 6
disc_env = create_reactor_env(use_dummy=False, n_dummy_actions=N_ACTIONS_TO_USE)



# --- QLearningAgent with Update Implemented ---
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

# --- Training Setup ---
agent = QLearningAgent(
    state_dim=disc_env.grid_size,
    action_dim=disc_env.n_actions,
    learning_rate=0.01,
    discount_factor=0.99,
    exploration_rate=1.0,
    exploration_decay=0.995,
    min_exploration_rate=0.001
)

# --- Training Loop ---
num_episodes = 2000
max_steps_per_episode = 15

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

# --- Visualization ---
value_function = np.max(agent.q_table, axis=2) # Shape: (num_ca_cells, num_temp_cells)
q_function = agent.q_table

initial_continuous_state_for_plot = None
try:
    ca_edges = disc_env.ca_bins
    temp_edges = disc_env.temp_bins
    goal_state_plot_val = disc_env.goal_state # Renamed to avoid conflict
    n_actions_val = disc_env.n_actions
    if hasattr(disc_env, 'x0_continuous'): # Check wrapper first
        initial_continuous_state_for_plot = disc_env.x0_continuous
    elif hasattr(disc_env, 'env') and hasattr(disc_env.env, 'x0_continuous_unnormalized'): # Check underlying env
        initial_continuous_state_for_plot = disc_env.env.x0_continuous_unnormalized

except AttributeError as e:
    print(f"Error getting plot parameters from environment: {e}. Using defaults.")
    ca_edges = np.linspace(0.7, 0.9, agent.state_dim[0] + 1)
    temp_edges = np.linspace(315., 335., agent.state_dim[1] + 1)
    goal_state_plot_val = 0.86
    n_actions_val = agent.action_dim

print(f"Visualization using n_actions = {n_actions_val}")

def visualize_value_function(v_func_ca_temp, ca_edges, temp_edges, goal_ca_val, ax):
    # v_func_ca_temp has shape (num_ca_cells, num_temp_cells)
    plot_extent = [temp_edges[0], temp_edges[-1], ca_edges[0], ca_edges[-1]]
    masked_v_func = np.ma.masked_where(v_func_ca_temp == 0, v_func_ca_temp)
    valid_values = masked_v_func.compressed()

    if valid_values.size > 0 and np.ptp(valid_values) > 1e-6:
         vmin = np.min(valid_values)
         vmax = np.max(valid_values)
    else:
        ref_val = np.mean(v_func_ca_temp) if v_func_ca_temp.size > 0 else 0.0
        vmin = ref_val - 0.1
        vmax = ref_val + 0.1
        if abs(vmin - vmax) < 1e-9 : vmax += 1e-6

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.PiYG
    cmap.set_bad(color='lightgrey')

    im = ax.imshow(masked_v_func, extent=plot_extent, aspect='auto', cmap=cmap, origin='lower', interpolation='nearest', norm=norm)
    goal_line = ax.axhline(y=goal_ca_val, color='red', linestyle='--', linewidth=2, label=f'Goal Ca ({goal_ca_val:.2f})')
    ax.set_title('State Value Function V(s) = max$_a$ Q(s,a)')
    ax.set_xlabel('Reactor Temperature (K)')
    ax.set_ylabel('Concentration Ca (mol/L)')
    ax.set_xlim(temp_edges[0], temp_edges[-1])
    ax.set_ylim(ca_edges[0], ca_edges[-1])
    ax.set_xticks(temp_edges)
    ax.set_yticks(ca_edges)
    ax.set_xticklabels([f"{t:.1f}" for t in temp_edges], rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels([f"{ca:.3f}" for ca in ca_edges], fontsize=8)
    ax.grid(True, which='major', color='black', linestyle='-', linewidth=0.4, alpha=0.4)
    plt.colorbar(im, ax=ax, label='State Value')
    return [goal_line]

def add_action_temperature_key(ax, num_actions_key, cmap_legend_name='coolwarm'):
    if num_actions_key <= 0: return
    cmap_legend = plt.get_cmap(cmap_legend_name)
    ex_x, ex_y, ex_w, ex_h = 0.30, 1.06, 0.40, 0.06
    strip_w = ex_w / num_actions_key
    ax.add_patch(Rectangle((ex_x, ex_y), ex_w, ex_h, fill=False, edgecolor='black', lw=1, clip_on=False, transform=ax.transAxes))
    ax.text(ex_x + ex_w / 2, ex_y + ex_h + 0.01, 'Action Key: Temperature Effect', ha='center', va='bottom', transform=ax.transAxes, fontsize=10, weight='bold')
    for i in range(num_actions_key):
        strip_x = ex_x + i * strip_w
        color_val = i / (num_actions_key - 1) if num_actions_key > 1 else 0.5
        ax.add_patch(Rectangle((strip_x, ex_y), strip_w, ex_h, facecolor=cmap_legend(color_val), edgecolor='grey', lw=0.5, clip_on=False, transform=ax.transAxes))
    ax.text(ex_x, ex_y - 0.01, 'Colder', ha='center', va='top', transform=ax.transAxes, fontsize=9)
    ax.text(ex_x + ex_w, ex_y - 0.01, 'Hotter', ha='center', va='top', transform=ax.transAxes, fontsize=9)
    ax.annotate('', xy=(ex_x + ex_w * 0.95, ex_y - 0.04), xytext=(ex_x + ex_w * 0.05 , ex_y - 0.04), arrowprops=dict(arrowstyle="<->", color='black'), annotation_clip=False, transform=ax.transAxes)

def visualize_q_function_pie_segments(q_func_full, ca_edges_q, temp_edges_q, goal_ca_q, n_actions_q, ax):
    num_ca_cells, num_temp_cells, num_actions_from_q = q_func_full.shape
    if n_actions_q != num_actions_from_q:
        print(f"Warning: n_actions provided ({n_actions_q}) differs from q_func_full.shape[2] ({num_actions_from_q}). Using {num_actions_from_q}.")
        n_actions_q = num_actions_from_q
    if n_actions_q <= 0: return ax, []

    q_finite = q_func_full[np.isfinite(q_func_full)]
    if q_finite.size == 0: q_min, q_max = -1.0, 1.0
    else:
        q_min, q_max = np.min(q_finite), np.max(q_finite)
        if abs(q_min - q_max) < 1e-9: q_min -= 0.5; q_max += 0.5
    norm = mcolors.Normalize(vmin=q_min, vmax=q_max)
    cmap_qval = plt.cm.PiYG
    default_color = 'lightgrey'

    for r in range(num_ca_cells):
        ca_low, ca_high = ca_edges_q[r], ca_edges_q[r+1]
        cell_height = ca_high - ca_low
        for c in range(num_temp_cells):
            temp_low, temp_high = temp_edges_q[c], temp_edges_q[c+1]
            cell_width = temp_high - temp_low
            if n_actions_q == 0: continue

            q_s_current = q_func_full[r, c, :]

            if np.all(q_s_current == 0):
                grey_cell_rect = Rectangle((temp_low, ca_low), cell_width, cell_height,
                                           facecolor=default_color, edgecolor='silver', linewidth=0.2, zorder=0)
                ax.add_patch(grey_cell_rect)
                continue

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
                
                edge_color_for_strip = 'dimgray'
                line_width_for_strip = 0.2
                z_order_for_strip = 1

                if action_idx == best_action_to_highlight and np.isfinite(q_val):
                    edge_color_for_strip = 'black'
                    line_width_for_strip = 1.0 # Thicker highlight
                    z_order_for_strip = 5      # Draw on top

                strip_x_low = temp_low + action_idx * strip_width
                rect = Rectangle((strip_x_low, ca_low), strip_width, cell_height,
                                 facecolor=color, edgecolor=edge_color_for_strip,
                                 linewidth=line_width_for_strip, zorder=z_order_for_strip)
                ax.add_patch(rect)

    goal_line = ax.axhline(y=goal_ca_q, color='red', linestyle='--', linewidth=2, label=f'Goal Ca ({goal_ca_q:.2f})', zorder=10)
    ax.set_xlim(temp_edges_q[0], temp_edges_q[-1])
    ax.set_ylim(ca_edges_q[0], ca_edges_q[-1])
    ax.set_xticks(temp_edges_q)
    ax.set_yticks(ca_edges_q)
    ax.set_xticklabels([f"{t:.1f}" for t in temp_edges_q], rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels([f"{ca:.3f}" for ca in ca_edges_q], fontsize=8)
    ax.grid(True, which='major', color='black', linestyle='-', linewidth=0.4, alpha=0.4, zorder=0) # Main grid behind strips
    ax.set_title(f'Q(s, a) - Action Strips (Best Action Highlighted)')
    ax.set_xlabel('Reactor Temperature (K)')
    ax.set_ylabel('Concentration Ca (mol/L)')

    sm = plt.cm.ScalarMappable(cmap=cmap_qval, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Action Value (Q-value)', fraction=0.046, pad=0.04)
    add_action_temperature_key(ax, n_actions_q, cmap_legend_name='coolwarm')
    standard_legend_handles = [goal_line]
    return ax, standard_legend_handles

# --- Create the Plots ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

v_handles = visualize_value_function(value_function, ca_edges, temp_edges, goal_state_plot_val, axes[0])
q_ax, q_standard_handles = visualize_q_function_pie_segments(q_function, ca_edges, temp_edges, goal_state_plot_val, n_actions_val, axes[1])

try:
    init_ca_val_plot, init_temp_val_plot = None, None
    if initial_continuous_state_for_plot is not None:
        init_ca_val_plot = initial_continuous_state_for_plot[0] # Ca is typically first element
        init_temp_val_plot = initial_continuous_state_for_plot[1] # Temp is typically second
        print(f"Plotting initial state marker from continuous: (Ca: {init_ca_val_plot:.3f}, Temp: {init_temp_val_plot:.1f})")
    else:
        temp_env_for_reset = create_reactor_env(use_dummy=True, n_dummy_actions=N_ACTIONS_TO_USE) # Create a fresh env for reset
        initial_obs_discrete, _ = temp_env_for_reset.reset(seed=42)
        temp_env_for_reset.close()
        init_ca_idx, init_temp_idx = initial_obs_discrete[0], initial_obs_discrete[1]
        init_ca_val_plot = (ca_edges[init_ca_idx] + ca_edges[init_ca_idx + 1]) / 2
        init_temp_val_plot = (temp_edges[init_temp_idx] + temp_edges[init_temp_idx + 1]) / 2
        print(f"Plotting initial state marker from discrete center: (Ca idx {init_ca_idx}, Temp idx {init_temp_idx})")

    plot_y_ca_center = np.clip(init_ca_val_plot, ca_edges[0], ca_edges[-1])
    plot_x_temp_center = np.clip(init_temp_val_plot, temp_edges[0], temp_edges[-1])

    start_marker_v, = axes[0].plot(plot_x_temp_center, plot_y_ca_center, 'p', markersize=12, markeredgecolor='k', markerfacecolor='yellow', label='Start State', zorder=10)
    start_marker_q, = axes[1].plot(plot_x_temp_center, plot_y_ca_center, 'p', markersize=12, markeredgecolor='k', markerfacecolor='yellow', label='Start State', zorder=15)

    v_handles.append(start_marker_v)
    q_standard_handles.append(start_marker_q)
    axes[0].legend(handles=v_handles, loc='upper right')
    axes[1].legend(handles=q_standard_handles, loc='upper right')
except Exception as e:
    print(f"Could not plot initial state marker or finalize legends: {e}")
    import traceback
    traceback.print_exc()
    if v_handles: axes[0].legend(handles=v_handles, loc='upper right')
    else: axes[0].legend(loc='upper right')
    if q_standard_handles: axes[1].legend(handles=q_standard_handles, loc='upper right')
    else: axes[1].legend(loc='upper right')

# plt.tight_layout(rect=[0, 0.03, 1, 0.90]) # rect=[left, bottom, right, top]
plt.subplots_adjust(top=0.88) # Further adjust top to ensure action key doesn't overlap figure title if any, or Q-plot title
plt.savefig('Q_V_func_plot.svg', bbox_inches='tight')
plt.show()

# Optional: Plot learning curve
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, alpha=0.6, label='Episode Reward')
window_size = 100
if len(episode_rewards) >= window_size:
    moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(window_size - 1, len(episode_rewards)), moving_avg, color='red', label=f'{window_size}-Ep Moving Avg')
elif len(episode_rewards) > 0 :
    num_rewards_total = len(episode_rewards)
    # Calculate average over all available episodes if less than window_size
    overall_avg = np.mean(episode_rewards)
    # Plot this single average value as a point, perhaps at the end or middle
    plt.plot(num_rewards_total - 1 , overall_avg, 'o', color='orange', markersize=8, label=f'Overall Avg Reward ({num_rewards_total} Eps)')

plt.title('Episode Rewards over Time')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
plt.show()