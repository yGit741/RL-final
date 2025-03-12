from typing import Any, Union, Optional, Tuple
from gymnasium import Env
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import os
from IPython.display import display, Video


def show_full_frame_rgb(
    env: Env,
    obs: Optional[np.ndarray] = None,
    fig_size: tuple[int, int] = (4, 4),
    title: Optional[str] = None,
    return_rgb: bool = False
) -> None:
    """
    Displays the current frame of the environment in RGB format.

    Parameters:
        env (Env): The Gymnasium environment instance
        obs (Optional[np.ndarray]): The current observation (not used in rendering,
                                  but kept for API consistency)
        fig_size (tuple[int, int]): Size of the figure to display (width, height)
        title (Optional[str]): Title to display above the frame

    Returns:
        None

    Raises:
        ValueError: If the environment doesn't support rendering
        TypeError: If the rendered frame is not a valid image array
    """
    try:
        # Get the current environment frame as an image (RGB array)
        img = env.render()
        
        if not isinstance(img, np.ndarray):
            raise TypeError("Environment render must return a numpy array")

        # Set up the figure
        plt.figure(figsize=fig_size)
        
        # Display the image
        plt.imshow(img)
        
        # Add title if provided
        if title:
            plt.title(title)
            
        # Remove axis labels for better visualization
        plt.axis("off")
        
        # Show the plot
        plt.show()
        plt.close()  # Clean up resources

        if return_rgb:
            return img
        
    except Exception as e:
        plt.close()  # Ensure figure is closed even if there's an error
        raise e

def show_partial_greyscale(
    env: Env,
    obs: np.ndarray,
    fig_size: tuple[int, int] = (4, 4),
    title: Optional[str] = None,
    cmap: str = "gray",
    return_grayscale_array: bool = False
) -> None:
    """
    Displays the partial visible frame in grayscale format.

    Parameters:
        env (Env): The environment instance (kept for API consistency)
        obs (np.ndarray): The current observation to display
        fig_size (tuple[int, int]): Size of the figure to display (width, height)
        title (Optional[str]): Title to display above the frame
        cmap (str): Colormap to use for grayscale display

    Returns:
        None

    Raises:
        ValueError: If the observation is None or invalid shape
        TypeError: If the observation is not a numpy array
    """
    try:
        # Input validation
        if not isinstance(obs, np.ndarray):
            raise TypeError("Observation must be a numpy array")
        
        if obs.size == 0:
            raise ValueError("Observation array is empty")

        # Set up the figure
        plt.figure(figsize=fig_size)
        
        # Convert RGB to grayscale if needed
        if len(obs.shape) == 3 and obs.shape[-1] == 3:
            frame_gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            frame_gray = obs
            
        # Display the grayscale image
        plt.imshow(frame_gray, cmap=cmap)
        
        # Add title if provided
        if title:
            plt.title(title)
            
        # Remove axis labels for better visualization
        plt.axis("off")
        
        # Show the plot
        plt.show()
        plt.close()  # Clean up resources
        if return_grayscale_array:
            return frame_gray  
        
    except Exception as e:
        plt.close()  # Ensure figure is closed even if there's an error
        raise e

def show_state_full_and_partial(env, obs):
    """
    Displays the full environment view and the partial view on eon top of another.

    Parameters:
        env: The environment instance.
        obs (numpy.ndarray): The observation from the environment.
    """
    # Get full environment render
    full_view = env.render()

    # Create subplot with original and rotated images
    fig, axes = plt.subplots(2, 1, figsize=(3, 6))  # Two views for better comparison
    axes[0].imshow(full_view)
    axes[0].set_title("Full Environment View")
    axes[0].axis("off")
    axes[1].imshow(obs, cmap="gray")
    axes[1].set_title("Partial View")
    axes[1].axis("off")
    plt.show()

def render_agent_video(
    env: Env,
    agent: Any,
    filename: str = "videos/agent_video.mp4",
    fps: int = 16,
    seed: Optional[int] = None,
    frame_size: Tuple[int, int] = (512, 512)
) -> None:
    """
    Runs the agent in the given MultiRoom environment and records a video.

    Parameters:
        env (Env): The environment instance (e.g., MultiRoom).
        agent (Any): The agent instance with a `select_action(state)` method.
        filename (str): Path to save the recorded video.
        fps (int): Frames per second for the video.
        seed (Optional[int]): Seed for reproducibility.
        frame_size (Tuple[int, int]): Desired frame size (width, height) for resizing.

    Returns:
        None

    Raises:
        ValueError: If the environment cannot be rendered.
        FileNotFoundError: If the video directory does not exist.
    """
    # Ensure the directory for the video exists
    video_dir = os.path.dirname(filename)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    print(f"Saving video to: {filename}")

    # Reset environment with optional seed
    if seed is not None:
        env.reset(seed=seed)

    truncated = False  
    total_reward = 0
    step = 0

    try:
        with imageio.get_writer(filename, fps=fps) as video:
            obs, _ = env.reset()
            done = False
            print("!!!!!!")
            while not truncated:
                print("here")
                action = agent.select_action(obs)  # Use agent's action selection
                print(action)
                obs, reward, done, truncated, _ = env.step(action)
                print(obs.shape)
                print("obs", obs)
                
                
                total_reward += reward

                # Render and resize the frame
                frame = env.render()
                if not isinstance(frame, np.ndarray):
                    raise ValueError("Environment render must return a numpy array")

                frame = cv2.resize(frame, frame_size)  # Resize frame
                print(frame)
                video.append_data(frame)

                step += 1
                if done:
                    print(f"Done | Total Reward: {total_reward} | Steps: {step}")
                    break
                elif truncated:
                    print(f"Truncated | Total Reward: {total_reward} | Steps: {step}")
                    break

        # Display the video if running in a notebook
        display(Video(filename, embed=True))

    except Exception as e:
        print(f"Error during video generation: {e}")



# def plot_training_process(training_process, window_size = 100):
#     """Plots training progress including multiple metrics."""
#     fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    
#     # Reward Trend
#     axes[0, 0].plot(training_process.rewards_log)
#     axes[0, 0].set_title("Total Reward per Episode")
#     axes[0, 0].set_xlabel("Episode")
#     axes[0, 0].set_ylabel("Reward")
    
#     # Loss Trend
#     axes[0, 1].plot(training_process.loss_log, color='r')
#     axes[0, 1].set_title("Loss per Episode")
#     axes[0, 1].set_xlabel("Episode")
#     axes[0, 1].set_ylabel("Loss")
    
#     # Epsilon Decay
#     axes[1, 0].plot(training_process.epsilon_history, color='g')
#     axes[1, 0].set_title("Epsilon Decay")
#     axes[1, 0].set_xlabel("Episode")
#     axes[1, 0].set_ylabel("Epsilon")
    
#     # Q-values Trend
#     axes[1, 1].plot(training_process.q_values_history, color='purple')
#     axes[1, 1].set_title("Q-values Over Time")
#     axes[1, 1].set_xlabel("Episode")
#     axes[1, 1].set_ylabel("Q-values")
    
#     # Gradient Norms
#     axes[2, 0].plot(training_process.grad_norms, color='orange')
#     axes[2, 0].set_title("Gradient Norms Over Time")
#     axes[2, 0].set_xlabel("Episode")
#     axes[2, 0].set_ylabel("Gradient Norm")
    
#     # Action Selection Frequency
#     actions = list(training_process.action_counts.keys())
#     counts = list(training_process.action_counts.values())
#     axes[2, 1].bar(actions, counts, color='gray')
#     axes[2, 1].set_title("Action Selection Frequency")
#     axes[2, 1].set_xlabel("Actions")
#     axes[2, 1].set_ylabel("Count")
#     axes[2, 1].set_xticks(actions)  # Set x-axis labels as the action names
#     axes[2, 1].set_xticklabels(actions, rotation=45)
    
#     # Episode Length Trend
#     axes[3, 0].plot(training_process.episode_lengths, color='blue')
#     axes[3, 0].set_title("Episode Length Over Time")
#     axes[3, 0].set_xlabel("Episode")
#     axes[3, 0].set_ylabel("Steps")

#     # Success Rate
#     success_avg = [np.mean(training_process.success_rate[max(0, i - window_size + 1):i + 1]) for i in range(len(training_process.success_rate))]
#     axes[3, 1].plot(range(len(success_avg)), success_avg, color='cyan')
#     axes[3, 1].set_title("Success Rate Over Time")
#     axes[3, 1].set_xlabel("Episode")
#     axes[3, 1].set_ylabel("Success Rate")

    
#     plt.tight_layout()
#     plt.show()



def plot_training_process(training_process: Any, window_size: int = 100) -> None:
    """
    Plots training progress including multiple metrics.

    Parameters:
        training_process (Any): An object containing training logs.
        window_size (int): Window size for computing the rolling success rate.

    Returns:
        None
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))

    # Reward Trend
    axes[0, 0].plot(training_process.rewards_log, label="Reward")
    axes[0, 0].set_title("Total Reward per Episode")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # Loss Trend
    axes[0, 1].plot(training_process.loss_log, color='r', label="Loss")
    axes[0, 1].set_title("Loss per Episode")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # Epsilon Decay
    axes[1, 0].plot(training_process.epsilon_history, color='g', label="Epsilon")
    axes[1, 0].set_title("Epsilon Decay")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Epsilon")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # Q-values Trend
    axes[1, 1].plot(training_process.q_values_history, color='purple', label="Q-values")
    axes[1, 1].set_title("Q-values Over Time")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Q-values")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    # Gradient Norms
    axes[2, 0].plot(training_process.grad_norms, color='orange', label="Gradient Norm")
    axes[2, 0].set_title("Gradient Norms Over Time")
    axes[2, 0].set_xlabel("Episode")
    axes[2, 0].set_ylabel("Gradient Norm")
    axes[2, 0].grid(True)
    axes[2, 0].legend()

    # Action Selection Frequency
    action_mapping = {0: "Left", 1: "Right", 2: "Forward", 5: "Toggle"}
    filtered_actions = {a: count for a, count in training_process.action_counts.items() if a in action_mapping}
    action_labels = [f"{a}-{action_mapping[a]}" for a in filtered_actions.keys()]
    action_counts = list(filtered_actions.values())
    axes[2, 1].bar(action_labels, action_counts, color='gray')
    axes[2, 1].set_title("Action Selection Frequency")
    axes[2, 1].set_xlabel("Actions")
    axes[2, 1].set_ylabel("Count")
    axes[2, 1].set_xticks(range(len(action_labels)))  # Set positions
    axes[2, 1].set_xticklabels(action_labels, rotation=45)  # Use formatted labels



    # Episode Length Trend
    axes[3, 0].plot(training_process.episode_lengths, color='blue', label="Episode Length")
    axes[3, 0].set_title("Episode Length Over Time")
    axes[3, 0].set_xlabel("Episode")
    axes[3, 0].set_ylabel("Steps")
    axes[3, 0].grid(True)
    axes[3, 0].legend()

    # Success Rate
    success_avg = [
        np.mean(training_process.success_rate[max(0, i - window_size + 1):i + 1])
        for i in range(len(training_process.success_rate))
    ]
    axes[3, 1].plot(range(len(success_avg)), success_avg, color='cyan', label="Success Rate")
    axes[3, 1].set_title("Success Rate Over Time")
    axes[3, 1].set_xlabel("Episode")
    axes[3, 1].set_ylabel("Success Rate")
    axes[3, 1].grid(True)
    axes[3, 1].legend()

    plt.tight_layout()
    plt.show()
