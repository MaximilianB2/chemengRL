import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, FancyArrowPatch


class DiscreteReactorWrapper(gym.Wrapper):
    """
    Wrapper to convert a continuous chemical reactor environment 
    into a discrete grid world environment.
    """
    def __init__(self, env, grid_size=(100, 100), goal_state=0.85):
        """
        Args:
            env: The continuous environment to wrap
            grid_size: Tuple (nx, ny, nz) defining the grid resolution for each dimension
            goal_state: The target state for the environment
        """
        self.goal_state = goal_state
        super().__init__(env)
        self.continuous_env = env
        
        # Define grid dimensions
        self.grid_size = grid_size
        
        # Define discrete action space
        # We'll map continuous action range to n discrete actions
        self.n_actions = 7  # For example, 7 discrete temperature settings
        self.action_space = spaces.Discrete(self.n_actions)
        
        # Define discrete observation space
        # 3D grid representing discretized state space (Ca, T, Cb)
        self.observation_space = spaces.MultiDiscrete(grid_size)
        
        # Store continuous bounds for conversion
        self.cont_action_low = env.env_params['a_space']['low'][0]
        self.cont_action_high = env.env_params['a_space']['high'][0]
        self.cont_obs_low = env.env_params['o_space']['low']
        self.cont_obs_high = env.env_params['o_space']['high']
        
        # Pre-compute action mapping
        self.action_map = np.linspace(
            self.cont_action_low, 
            self.cont_action_high, 
            self.n_actions
        )
        
        # Define colormaps and visualization settings
        self.setup_visualization()
    
    def setup_visualization(self):
        """Setup visualization parameters"""
        # Color map for different dimensions
        self.cmap = plt.cm.viridis
        
        # For reward visualization
        self.reward_cmap = plt.cm.RdYlGn
        
        # Grid cell colors to visualize states
        self.ca_color = 'blue'      # Concentration A 
        self.t_color = 'red'        # Temperature
        self.cb_color = 'green'     # Concentration B
        
        # Trail colors - using a colormap for the trajectory
        self.trail_cmap = plt.cm.cool
        
        # Action colors based on value
        self.action_cmap = plt.cm.plasma
        
        # Save state and action history for visualization
        self.state_history = []
        self.action_history = []
        self.reward_history = []

    def continuous_to_discrete_obs(self, cont_obs):
        """Convert continuous observation to discrete grid coordinates"""
        
        # Scale the continuous observation from [-1, 1] to the observation space bounds
        scaled_obs = 0.5 * (cont_obs + 1) * (self.cont_obs_high - self.cont_obs_low) + self.cont_obs_low

        # Divide the observation space into 10 bins for each dimension
        bins = [np.linspace(low, high, num=11) for low, high in zip(self.cont_obs_low, self.cont_obs_high)]
        
        # Assign the continuous observation to the corresponding bin index
        discrete_obs = [np.digitize(value, bins[i]) - 1 for i, value in enumerate(scaled_obs[:-1])]
        
        # Clip to ensure values stay within bounds
        discrete_obs = np.clip(discrete_obs, 0, np.array(self.grid_size) - 1)
        
        return np.array(discrete_obs)
    
    def discrete_to_continuous_action(self, discrete_action):
        """Convert discrete action index to continuous action value"""
        return np.array([self.action_map[discrete_action]], dtype=np.float32)
    
    def reset(self, **kwargs):
        """Reset the environment and return discretized observation"""
        obs, info = self.env.reset(**kwargs)
        
        # Convert continuous observation to discrete
        discrete_obs = self.continuous_to_discrete_obs(obs)
        
        # Reset history
        self.state_history = [discrete_obs]
        self.action_history = []
        self.reward_history = []
        
        return discrete_obs, info
    
    def step(self, discrete_action):
        """Take a step using a discrete action"""
        # Convert discrete action to continuous
        continuous_action = self.discrete_to_continuous_action(discrete_action)
        # Normalize the continuous action to the range [-1, 1]
        continuous_action = 2 * (continuous_action - self.cont_action_low) / (self.cont_action_high - self.cont_action_low) - 1
        # Step in the continuous environment
        obs, reward, terminated, truncated, info = self.env.step(continuous_action)
        
        # Convert observation to discrete
        discrete_obs = self.continuous_to_discrete_obs(obs)
        
        # Store for visualization
        self.state_history.append(discrete_obs)
        self.action_history.append(discrete_action)
        self.reward_history.append(reward)
        
        return discrete_obs, reward, terminated, truncated, info
    
    def render_grid(self, ax=None, step_idx=-1, show_full_trajectory=False, value_function=None):
        """Render current state as a 2D grid with improved visualization"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        
        # Create grid layout
        grid = np.zeros((self.grid_size[0], self.grid_size[1]))
        
        # Get current state
        state = self.state_history[step_idx]
        ca_idx, t_idx = state
        
        # Clear the plot first
        ax.clear()
        
        # Draw the grid with lighter background
        ax.imshow(grid, cmap='binary', alpha=0.1, vmin=0, vmax=1)
        
        # Draw the goal region - highlight the entire row
        # Create a horizontal line across the full width for the goal
        # Convert the goal state to the corresponding grid index
        goal_y = int((self.goal_state - self.cont_obs_low[0]) / 
            (self.cont_obs_high[0] - self.cont_obs_low[0]) * (self.grid_size[0] - 1))
        
        # Draw a horizontal line at the goal state
        ax.axhline(y=goal_y, color='cyan', linestyle='--', alpha=0.5, linewidth=2)
        
        # Add a label for the goal state
        ax.text(self.grid_size[1]-1, goal_y, "Goal", 
        color='cyan', ha='right', va='center', 
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        if value_function is not None:
            if value_function.shape != self.grid_size:
                raise ValueError(f"Value function must match grid size {self.grid_size}, got {value_function.shape}")
            
            # Mask zero values to avoid plotting them
            masked_value_function = np.ma.masked_where(value_function == 0, value_function)
            
            norm = mcolors.Normalize(vmin=np.min(masked_value_function), vmax=np.max(masked_value_function))
            heatmap = ax.imshow(masked_value_function, cmap='viridis', norm=norm, alpha=0.5, origin='lower')
            cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Value Function', fontsize=12)

        # Display the trajectory
        if show_full_trajectory:
            # Use all states for the full trajectory in the static image
            trail_states = self.state_history
        else:
            # Determine how many previous states to show (adjust based on step_idx)
            trail_length = min(10, step_idx + 1)
            # Extract the trail states
            trail_states = self.state_history[max(0, step_idx-trail_length+1):step_idx+1]
        
        if len(trail_states) > 1:
            # Extract trail coordinates
            trail_x = [s[1] for s in trail_states]
            trail_y = [s[0] for s in trail_states]
            
            # Plot the full trajectory line with no fading
            ax.plot(trail_x, trail_y, 'k-', alpha=0.9, linewidth=2, zorder=4)
            
            # Plot all trail points with consistent color and size (no fading)
            for i, prev_state in enumerate(trail_states):
                # Use a fixed color for all trail points or vary by position
                if show_full_trajectory:
                    # For full trajectory, color based on time
                    color_val = i / (len(trail_states) - 1) if len(trail_states) > 1 else 1
                    color = self.trail_cmap(color_val)
                else:
                    # For animation trail, use a single color for consistency
                    color = 'blue'
                
                # Use consistent size for all points
                size = 80
                
                # Plot each point with full opacity
                ax.scatter(prev_state[1], prev_state[0], c=[color], s=size, alpha=1.0,
                        edgecolors='black', linewidth=1, zorder=5)
            
            # Add direction arrows (except for the static full trajectory which would be too cluttered)
            if not show_full_trajectory and len(trail_states) >= 2:
                for i in range(len(trail_states)-1):
                    dx = trail_x[i+1] - trail_x[i]
                    dy = trail_y[i+1] - trail_y[i]
                    if abs(dx) > 0.1 or abs(dy) > 0.1:  # Only add arrow if there's movement
                        ax.arrow(trail_x[i], trail_y[i], dx*0.8, dy*0.8, 
                                head_width=0.2, head_length=0.3, fc='black', ec='black', 
                                alpha=0.9, zorder=4)
        
        # Mark the current state with a distinctive marker
        ax.scatter(t_idx, ca_idx, c='red', s=250, marker='*', edgecolors='white',
                linewidth=1.5, label='Current State', zorder=20)
        
        # Add text label with current state values
        ca_val = self.cont_obs_low[0] + ca_idx * (self.cont_obs_high[0] - self.cont_obs_low[0]) / (self.grid_size[0] - 1)
        t_val = self.cont_obs_low[1] + t_idx * (self.cont_obs_high[1] - self.cont_obs_low[1]) / (self.grid_size[1] - 1)
        state_text = f"Ca={ca_val:.3f}, T={t_val:.1f}K"
        ax.text(t_idx, ca_idx-0.5, state_text, color='white', ha='center', va='top',
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, self.grid_size[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_size[0], 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)
        
        # Configure axes with ticks for each box
        ax.set_xticks(np.arange(0, self.grid_size[1], 1))
        ax.set_yticks(np.arange(0, self.grid_size[0], 1))
        
        # Label axes with actual continuous values - ensure 10 ticks to match grid
        t_ticks = np.linspace(self.cont_obs_low[1], self.cont_obs_high[1], self.grid_size[1])
        ca_ticks = np.linspace(self.cont_obs_low[0], self.cont_obs_high[0], self.grid_size[0])
        
        # Position ticks at each grid cell center
        t_positions = np.arange(0, self.grid_size[1])
        ca_positions = np.arange(0, self.grid_size[0])
        
        ax.set_xticks(t_positions)
        ax.set_yticks(ca_positions)
        ax.set_xticklabels([f"{t:.1f}" for t in t_ticks])
        ax.set_yticklabels([f"{ca:.2f}" for ca in ca_ticks])
        
        # Set labels
        ax.set_xlabel("Temperature (K)", fontweight='bold')
        ax.set_ylabel("Concentration Ca", fontweight='bold')
        
        # Set appropriate title based on mode
        if show_full_trajectory:
            ax.set_title(f"Chemical Reactor Grid World - Full Trajectory ({len(self.state_history)} steps)", 
                    fontsize=14, fontweight='bold')
        else:
            ax.set_title(f"Chemical Reactor Grid World - Step {step_idx}", 
                    fontsize=14, fontweight='bold')
        
        # Add actions information if available with improved visualization
        if step_idx > 0 and len(self.action_history) >= step_idx:
            action_idx = self.action_history[step_idx-1]
            action_value = self.action_map[action_idx]
            reward = self.reward_history[step_idx-1]
            
            # Create an action panel at the bottom
            action_color = self.action_cmap(action_idx / (self.n_actions - 1))
            action_box = Rectangle((0, -2), self.grid_size[1]-1, 1.5, 
                                facecolor=action_color, alpha=0.3,
                                transform=ax.transData)
            ax.add_patch(action_box)
            
            # Add action text with more details
            reward_text = f"+" if reward > 0 else ""
            action_text = f"Action: {action_idx} (T={action_value:.2f}K)\nReward: {reward_text}{reward:.4f}"
            ax.text(self.grid_size[1]/2, -1.25, action_text, 
                transform=ax.transData, ha='center', va='center', 
                fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
            
            # Add action bar to show spectrum of possible actions
            action_bar_height = 0.3
            for a in range(self.n_actions):
                ax.add_patch(Rectangle((a * (self.grid_size[1]-1)/(self.n_actions), -2.7), 
                                    (self.grid_size[1]-1)/self.n_actions, action_bar_height, 
                                    facecolor=self.action_cmap(a/(self.n_actions-1)), 
                                    alpha=0.7))
            
            # Mark the current action on the action bar
            ax.scatter(action_idx * (self.grid_size[1]-1)/(self.n_actions) + 
                    (self.grid_size[1]-1)/(self.n_actions*2), 
                    -2.7 + action_bar_height/2, 
                    marker='v', s=150, color='white', edgecolors='black', zorder=10)
            
            # Add labels for action bar
            ax.text(0, -2.7 - action_bar_height/2, f"{self.cont_action_low:.1f}K", 
                ha='left', va='center', fontsize=9)
            ax.text(self.grid_size[1]-1, -2.7 - action_bar_height/2, f"{self.cont_action_high:.1f}K", 
                ha='right', va='center', fontsize=9)
    
        # Add legend with improved formatting
        legend = ax.legend(loc='upper right', framealpha=0.9, frameon=True, 
                        facecolor='white', edgecolor='gray')
        
        # Adjust axis to accommodate the action panel
        if not show_full_trajectory:
            ax.set_ylim(-3, self.grid_size[0]-1.5)
        else:
            ax.set_ylim(-1, self.grid_size[0]-1)
        return ax
    
    def visualize_rollout(self, save_path=None, value_function=None):
        """Visualize a complete rollout of the environment"""
        n_steps = len(self.state_history)
        
        if n_steps <= 1:
            print("No steps taken yet. Run a rollout first.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 🔥 Pass value_function into render_grid
        self.render_grid(ax=ax, step_idx=-1, show_full_trajectory=True, value_function=value_function)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()

    
    def animate_rollout(self, save_path=None, interval=500):
        """Create an animation of the environment rollout and save it to file without displaying"""
        n_steps = len(self.state_history)
        
        if n_steps <= 1:
            print("No steps taken yet. Run a rollout first.")
            return
        
        # Create figure without displaying it
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.ioff()  # Turn off interactive mode
        
        def animate(i):
            ax.clear()
            self.render_grid(ax=ax, step_idx=i)
            return [ax]
        
        ani = FuncAnimation(fig, animate, frames=range(n_steps),
                        interval=interval, blit=False)
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            ani.save(save_path, writer='pillow', fps=6, dpi=100)
            plt.close(fig)  # Close the figure after saving
            print(f"Animation saved successfully.")
            return True
        else:
            plt.close(fig)  # Close the figure if not saving
            print("No save path provided. Animation was created but not saved.")
            return False
