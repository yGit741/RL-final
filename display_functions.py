# from typing import Any, Union, Optional
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# from gymnasium import Env

def show_full_frame_rgb(
    env: Env,
    obs: Optional[np.ndarray] = None,
    fig_size: tuple[int, int] = (8, 8),
    title: Optional[str] = None
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
        # return img
        
    except Exception as e:
        plt.close()  # Ensure figure is closed even if there's an error
        raise e

def show_partial_greyscale(
    env: Env,
    obs: np.ndarray,
    fig_size: tuple[int, int] = (8, 8),
    title: Optional[str] = None,
    cmap: str = "gray"
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
