from typing import Tuple, Optional, Dict, Union
from gymnasium import Env
from minigrid.envs import MultiRoomEnv
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import numpy as np

def create_multiroom_env(
    num_rooms: int = 2,
    max_room_size: int = 5,
    render_mode: str = "rgb_array",
    seed: Optional[int] = None,
    max_steps: Optional[int] = None  ) -> Tuple[Env, np.ndarray]:
    """
    Create a wrapped MiniGrid MultiRoom environment.

    Parameters:
        num_rooms (int): Number of rooms in the environment.
        max_room_size (int): Maximum size of each room.
        render_mode (str): How to render the environment (default "rgb_array").
        seed (int, optional): Seed for environment randomness.

    Returns:
        Tuple[Env, np.ndarray]: A tuple containing:
            - env: The wrapped Gymnasium environment
            - obs: Initial observation from the environment
    
    Raises:
        ValueError: If num_rooms or max_room_size are less than 1
    """
    # Input validation
    if num_rooms < 1:
        raise ValueError("num_rooms must be at least 1")
    if max_room_size < 1:
        raise ValueError("max_room_size must be at least 1")

    # Default truncation limit if not specified
    if max_steps is None:
        max_steps = 20 * num_rooms**2  # Default truncation rule
    
    env = MultiRoomEnv(
        minNumRooms=num_rooms, 
        maxNumRooms=num_rooms, 
        maxRoomSize=max_room_size, 
        render_mode=render_mode
    )

    # Apply the requested wrappers
    env = RGBImgPartialObsWrapper(env)  # Converts observations to partial RGB images
    env = ImgObsWrapper(env)  # Ensures the observation space is image-based

    # Set the maximum number of steps    
    env.unwrapped.max_steps = max_steps

    # Reset environment with optional seed
    obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
    return env, obs

def get_action_meaning(action: Union[int, np.integer]) -> str:
    """
    Maps action numbers to their meanings.
    
    Parameters:
        action (Union[int, np.integer]): The action number to map
        
    Returns:
        str: The string description of the action
    """
    action_map: Dict[int, str] = {
        0: "Move Left",
        1: "Move Right",
        2: "Move Forward",
        5: "Toggle"
    }
    return action_map.get(int(action), "Unknown Action")


