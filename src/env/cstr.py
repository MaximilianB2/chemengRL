import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# import seaborn as sns # Not used in the provided snippet, can be removed if not needed elsewhere
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle # FancyArrowPatch not used, can remove

class DiscreteReactorWrapper(gym.Wrapper):
    """
    Wrapper to convert a continuous chemical reactor environment
    into a discrete grid world environment.
    """
    def __init__(self, env, grid_size=(10, 10), goal_state=0.85): # Defaulted grid_size to (10,10) for typical Q-learning
        """
        Args:
            env: The continuous environment to wrap
            grid_size: Tuple (num_ca_cells, num_temp_cells) defining the grid resolution.
            goal_state: The target state for the Ca concentration.
        """
        super().__init__(env)
        self.continuous_env = env
        self.goal_state = goal_state
        self.grid_size = grid_size # (num_ca_cells, num_temp_cells)

        self.n_actions = 7  # Example: 7 discrete temperature settings
        self.action_space = spaces.Discrete(self.n_actions)

        # Discrete observation space: [Ca_index, Temp_index]
        self.observation_space = spaces.MultiDiscrete(np.array(grid_size))

        self.cont_action_low = env.env_params['a_space']['low'][0]
        self.cont_action_high = env.env_params['a_space']['high'][0]
        self.cont_obs_low = env.env_params['o_space']['low']    # [Ca_low, T_low, Cb_low]
        self.cont_obs_high = env.env_params['o_space']['high']  # [Ca_high, T_high, Cb_high]

        # Define bin edges for Ca (concentration) and T (temperature)
        # Ca is the first dimension (y-axis), T is the second dimension (x-axis)
        self.ca_bins = np.linspace(self.cont_obs_low[0], self.cont_obs_high[0], self.grid_size[0] + 1)
        self.temp_bins = np.linspace(self.cont_obs_low[1], self.cont_obs_high[1], self.grid_size[1] + 1)

        self.action_map = np.linspace(
            self.cont_action_low,
            self.cont_action_high,
            self.n_actions
        )

        self.setup_visualization()

    def setup_visualization(self):
        self.cmap = plt.cm.viridis
        self.reward_cmap = plt.cm.RdYlGn
        self.ca_color = 'blue'
        self.t_color = 'red'
        self.cb_color = 'green' # Not used in 2D grid
        self.trail_cmap = plt.cm.cool
        self.action_cmap = plt.cm.plasma
        self.state_history = []
        self.action_history = []
        self.reward_history = []

    def continuous_to_discrete_obs(self, cont_obs):
        """Convert continuous observation (Ca, T) to discrete grid coordinates (ca_idx, t_idx)"""
        # Assuming cont_obs from env is normalized [-1, 1]
        # Scale normalized obs back to original continuous space
        # We only care about Ca (obs[0]) and T (obs[1]) for the 2D grid
        ca_cont_norm, t_cont_norm = cont_obs[0], cont_obs[1]

        ca_val = 0.5 * (ca_cont_norm + 1) * (self.cont_obs_high[0] - self.cont_obs_low[0]) + self.cont_obs_low[0]
        t_val = 0.5 * (t_cont_norm + 1) * (self.cont_obs_high[1] - self.cont_obs_low[1]) + self.cont_obs_low[1]
        
        # Digitize using pre-computed bins
        ca_idx = np.digitize(ca_val, self.ca_bins) - 1
        t_idx = np.digitize(t_val, self.temp_bins) - 1

        # Clip to ensure indices are within [0, grid_size-1]
        # np.digitize can return grid_size[i] if value is on self.cont_obs_high[i]
        ca_idx = np.clip(ca_idx, 0, self.grid_size[0] - 1)
        t_idx = np.clip(t_idx, 0, self.grid_size[1] - 1)

        return np.array([ca_idx, t_idx], dtype=int)

    def discrete_to_continuous_action(self, discrete_action):
        return np.array([self.action_map[discrete_action]], dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        discrete_obs = self.continuous_to_discrete_obs(obs)
        self.state_history = [discrete_obs]
        self.action_history = []
        self.reward_history = []
        return discrete_obs, info

    def step(self, discrete_action):
        continuous_action_val = self.discrete_to_continuous_action(discrete_action)
        # Normalize action for the underlying env if it expects normalized actions
        normalized_continuous_action = 2 * (continuous_action_val - self.cont_action_low) / \
                                   (self.cont_action_high - self.cont_action_low) - 1
        
        obs, reward, terminated, truncated, info = self.env.step(normalized_continuous_action)
        discrete_obs = self.continuous_to_discrete_obs(obs)

        self.state_history.append(discrete_obs)
        self.action_history.append(discrete_action)
        self.reward_history.append(reward)
        # Calculate reward based on discrete observation
        ca_idx, t_idx = discrete_obs
        ca_val, t_val = self._get_continuous_cell_center(ca_idx, t_idx)

        # Example reward calculation: proximity to goal state
        reward = -(ca_val - self.goal_state)**2
        return discrete_obs, reward, terminated, truncated, info

    def _get_continuous_cell_center(self, ca_idx, t_idx):
        """Get continuous coordinates of the center of a discrete cell."""
        ca_val = (self.ca_bins[ca_idx] + self.ca_bins[ca_idx+1]) / 2
        t_val = (self.temp_bins[t_idx] + self.temp_bins[t_idx+1]) / 2
        return ca_val, t_val

    def render_grid(self, ax=None, step_idx=-1, show_full_trajectory=False, value_function=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8)) # Adjusted for potentially taller Ca axis

        ax.clear()

        # Define continuous extents for imshow
        # extent = [temp_min, temp_max, ca_min, ca_max]
        plot_extent = [self.temp_bins[0], self.temp_bins[-1], self.ca_bins[0], self.ca_bins[-1]]

        # Draw background (optional, can be removed if value function covers all)
        # ax.imshow(np.zeros(self.grid_size), cmap='binary', alpha=0.1, origin='lower', extent=plot_extent)

        # Value Function Heatmap
        if value_function is not None:
            if value_function.shape != tuple(self.grid_size): # value_function shape (num_ca, num_temp)
                raise ValueError(f"Value function must match grid size {self.grid_size}, got {value_function.shape}")
            
            masked_value_function = np.ma.masked_where(value_function == 0, value_function) # Or other condition
            
            # Handle case where all values are masked or identical
            valid_values = masked_value_function[~masked_value_function.mask] if hasattr(masked_value_function, 'mask') else masked_value_function.flatten()
            if len(valid_values) > 0 :
                vmin = np.min(valid_values) if len(valid_values) > 0 else 0
                vmax = np.max(valid_values) if len(valid_values) > 0 else 1
                if vmin == vmax: # Avoid error in Normalize if all values are same
                    vmin -= 0.1
                    vmax += 0.1
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            else: # all masked
                norm = mcolors.Normalize(vmin=0, vmax=1)


            heatmap = ax.imshow(masked_value_function, cmap='viridis', norm=norm, alpha=0.7, origin='lower', extent=plot_extent, aspect='auto')
            cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('State Value V(s)', fontsize=12)

        # Goal Line (Horizontal line at goal Ca concentration)
        ax.axhline(y=self.goal_state, color='cyan', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(self.temp_bins[-1], self.goal_state, f" Goal Ca ({self.goal_state:.2f})",
                color='cyan', ha='right', va='center', backgroundcolor='white',alpha=0.7,
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))


        # Current state (indices and continuous center)
        ca_idx, t_idx = self.state_history[step_idx]
        current_ca_val, current_t_val = self._get_continuous_cell_center(ca_idx, t_idx)

        # Trajectory
        trail_states_indices = self.state_history if show_full_trajectory else \
                               self.state_history[max(0, step_idx - 9):step_idx + 1]
        
        if len(trail_states_indices) > 1:
            trail_ca_cont = []
            trail_t_cont = []
            for s_idx_pair in trail_states_indices:
                s_ca, s_t = self._get_continuous_cell_center(s_idx_pair[0], s_idx_pair[1])
                trail_ca_cont.append(s_ca)
                trail_t_cont.append(s_t)

            ax.plot(trail_t_cont, trail_ca_cont, 'k-', alpha=0.9, linewidth=2, zorder=4) # Temp on X, Ca on Y

            for i in range(len(trail_t_cont)):
                color_val = i / (len(trail_t_cont) - 1) if len(trail_t_cont) > 1 else 1
                color = self.trail_cmap(color_val) if show_full_trajectory else 'blue'
                ax.scatter(trail_t_cont[i], trail_ca_cont[i], c=[color], s=80, alpha=1.0,
                           edgecolors='black', linewidth=1, zorder=5)
            
            # Simplified arrows for animation
            if not show_full_trajectory and len(trail_t_cont) >= 2:
                for i in range(len(trail_t_cont)-1):
                    dx = trail_t_cont[i+1] - trail_t_cont[i]
                    dy = trail_ca_cont[i+1] - trail_ca_cont[i]
                    # Only add arrow if there's significant movement
                    if np.sqrt(dx**2 + dy**2) > 1e-3 * np.mean([self.temp_bins[-1]-self.temp_bins[0], self.ca_bins[-1]-self.ca_bins[0]]): # Heuristic for significance
                        ax.arrow(trail_t_cont[i], trail_ca_cont[i], dx*0.8, dy*0.8,
                                 head_width=(self.ca_bins[-1]-self.ca_bins[0])*0.02, # Relative head width
                                 head_length=(self.temp_bins[-1]-self.temp_bins[0])*0.02, # Relative head length
                                 fc='black', ec='black', alpha=0.9, zorder=4)

        # Mark current state
        ax.scatter(current_t_val, current_ca_val, c='red', s=250, marker='*', edgecolors='white',
                   linewidth=1.5, label='Current State', zorder=20)
        
        # State text label (using actual continuous values derived for plotting)
        # These are cell centers, for actual value use env.x
        # For label, use the center of the cell
        actual_ca_val_for_label, actual_t_val_for_label = self.continuous_env.x[0], self.continuous_env.x[1] # from underlying env if available
        state_text_current = f"Ca={actual_ca_val_for_label:.3f}\nT={actual_t_val_for_label:.1f}K"
        ax.text(current_t_val, current_ca_val - (self.ca_bins[1]-self.ca_bins[0])*0.8, state_text_current, # Adjusted position
                color='white', ha='center', va='top',
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.2'), zorder=21)

        # Axis grid, limits, and labels (using continuous values)
        ax.set_xlim(self.temp_bins[0], self.temp_bins[-1])
        ax.set_ylim(self.ca_bins[0], self.ca_bins[-1])

        ax.set_xticks(self.temp_bins)
        ax.set_yticks(self.ca_bins)
        ax.grid(which="major", color="gray", linestyle=':', linewidth=0.5)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        
        # Fewer ticks for readability if too many bins
        if len(self.temp_bins) > 15:
             ax.set_xticks(np.linspace(self.temp_bins[0], self.temp_bins[-1], 7))
        if len(self.ca_bins) > 15:
             ax.set_yticks(np.linspace(self.ca_bins[0], self.ca_bins[-1], 7))


        ax.set_xlabel("Reactor Temperature (K)", fontweight='bold')
        ax.set_ylabel("Concentration Ca (mol/L)", fontweight='bold') # Updated label
        ax.set_title(f"Reactor State - {'Full Trajectory' if show_full_trajectory else f'Step {step_idx}'}",
                     fontsize=14, fontweight='bold')

        # Action Panel (adjust y-positioning carefully relative to ca_bins[0])
        action_panel_height_abs = (self.ca_bins[-1] - self.ca_bins[0]) * 0.1 # Relative height
        action_panel_y_bottom = self.ca_bins[0] - action_panel_height_abs * 1.5 # Position below main plot

        if not show_full_trajectory and step_idx > 0 and len(self.action_history) >= step_idx:
            action_idx = self.action_history[step_idx-1]
            action_value = self.action_map[action_idx]
            reward = self.reward_history[step_idx-1]
            
            action_color = self.action_cmap(action_idx / (self.n_actions - 1))
            action_box_width = self.temp_bins[-1] - self.temp_bins[0]
            action_box = Rectangle((self.temp_bins[0], action_panel_y_bottom - action_panel_height_abs*0.2), # Slightly lower for action bar
                                    action_box_width, action_panel_height_abs,
                                    facecolor=action_color, alpha=0.3, transform=ax.transData, clip_on=False)
            ax.add_patch(action_box)
            
            reward_text = f"{'+' if reward > 0 else ''}{reward:.3f}"
            action_text = f"Action: {action_idx} (T_cool={action_value:.1f}K) | Reward: {reward_text}"
            ax.text((self.temp_bins[0] + self.temp_bins[-1])/2, action_panel_y_bottom + action_panel_height_abs/2 - action_panel_height_abs*0.2, action_text,
                    ha='center', va='center', fontsize=10, fontweight='bold', transform=ax.transData,
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

            # Action bar spectrum
            action_bar_element_width = action_box_width / self.n_actions
            action_bar_height_small = action_panel_height_abs * 0.2
            action_bar_y = action_panel_y_bottom - action_panel_height_abs * 0.2 - action_bar_height_small * 1.5

            for a_i in range(self.n_actions):
                ax.add_patch(Rectangle((self.temp_bins[0] + a_i * action_bar_element_width, action_bar_y),
                                        action_bar_element_width, action_bar_height_small,
                                        facecolor=self.action_cmap(a_i/(self.n_actions-1)), alpha=0.7,
                                        transform=ax.transData, clip_on=False))
            ax.scatter(self.temp_bins[0] + action_idx * action_bar_element_width + action_bar_element_width/2,
                       action_bar_y + action_bar_height_small/2,
                       marker='v', s=100, color='white', edgecolors='black', zorder=10, transform=ax.transData, clip_on=False)
            
            ax.text(self.temp_bins[0], action_bar_y - action_bar_height_small, f"{self.cont_action_low:.1f}K",
                    ha='left', va='center', fontsize=8, transform=ax.transData, clip_on=False)
            ax.text(self.temp_bins[-1], action_bar_y - action_bar_height_small, f"{self.cont_action_high:.1f}K",
                    ha='right', va='center', fontsize=8, transform=ax.transData, clip_on=False)

            # Adjust ylim to make space for action panel if it's drawn
            ax.set_ylim(action_bar_y - action_bar_height_small, self.ca_bins[-1]) # Ensure bottom of action bar is visible
        
        ax.legend(loc='upper right', framealpha=0.9, facecolor='white', edgecolor='gray')
        plt.tight_layout() # Call this before returning ax usually
        return ax

    def visualize_rollout(self, save_path=None, value_function=None):
        if len(self.state_history) <= 1:
            print("No steps taken yet. Run a rollout first.")
            return
        fig, ax = plt.subplots(figsize=(12, 10)) # Adjust size as needed
        self.render_grid(ax=ax, step_idx=-1, show_full_trajectory=True, value_function=value_function)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def animate_rollout(self, save_path=None, interval=500, value_function=None):
        if len(self.state_history) <= 1:
            print("No steps taken yet. Run a rollout first.")
            return
        fig, ax = plt.subplots(figsize=(10, 10)) # Adjust size as needed
        plt.ioff()
        
        def animate(i):
            # ax.clear() # render_grid clears its own axis
            self.render_grid(ax=ax, step_idx=i, show_full_trajectory=False, value_function=value_function)
            # Title is set in render_grid
            return [ax] # Should return list of artists changed
        
        ani = FuncAnimation(fig, animate, frames=len(self.state_history),
                            interval=interval, blit=False) # Blit=False is often more robust
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            ani.save(save_path, writer='pillow', fps=max(1, 1000//interval), dpi=100) # fps from interval
            plt.close(fig)
            print("Animation saved successfully.")
            return True
        else:
            # If not saving, you might want to display it interactively
            # For non-interactive environments or saving, plt.show() might not be desired here.
            # For now, just close if not saving.
            plt.close(fig)
            print("No save path provided. Animation was created but not saved.")
            return False