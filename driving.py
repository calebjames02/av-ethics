import gymnasium as gym
import highway_env
import os
import time
from graphs import graph_plot, graph_plot_point
from llm import ask_llm, prompt
from PIL import Image
from settings import load_settings, save_settings
from tools import closest_same_lane, Vehicle
from torch.utils.tensorboard import SummaryWriter

config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": True,
        "normalize": False
    }
}

# If disabled, default action of IDLE taken at each timestep
use_llm = False

# Number of episodes to run
num_episodes = 2

# Initialize variables for logging
episode_speeds, timesteps = ([] for _ in range(2))
output_folder = f"frames/run_{time.time()}/"

# Execute the given number of episodes
for episode in range(1, num_episodes + 1):
    env = gym.make('highway-v0', render_mode="rgb_array", config=config)

    state, _ = env.reset()
    frame_count = 0
    obs = state
    done = False
    speeds = []

    episode_folder = output_folder + f"episode_{episode}"

    # Create folder to save each frame of the environment to
    os.makedirs(episode_folder, exist_ok=True)

    # Create file to log output to
    log = open(f"{episode_folder}/log.txt", "w")

    while not done:
        # Render frame and save it locally if enabled
        frame = env.render()
        Image.fromarray(frame).save(f"{episode_folder}/frame_{frame_count:05d}.png")

        cars = [Vehicle(name="Ego vehicle", x_pos=round(obs[0][1], 4), lane=round(obs[0][2] / 4 + 1), x_vel=round(obs[0][3], 4), y_vel=round(obs[0][4], 4))]
        speeds.append(round(obs[0][3], 4))
        for i in range (1, len(obs)):
            cars.append(Vehicle(name=f"Vehicle: {i}", x_pos=round(obs[i][1], 4), lane=round(obs[i][2] / 4 + 1), x_vel=round(obs[i][3], 4), y_vel=round(obs[i][4], 4)))

        # Initialize log with output of all cars
        log.write(f"Timestep: {frame_count}\n\n")
        frame_count += 1
        for i in range (0, len(cars)):
            log.write(f"{cars[i]}\n")
        log.write("\n")

        if use_llm:
            available = env.unwrapped.action_type.get_available_actions()
            available_actions = [f"{action}: {env.unwrapped.action_type.ACTIONS_ALL[action]}" for action in available]

            # Action indeces don't match their position in the list which can be confusing for the LLM
            # Sorting the list ensures that the index in the string matches its position in the list
            available_actions.sort()

            response, action = ask_llm("gpt-oss", prompt, available_actions, cars)

            # Check that LLM hasn't returned an invalid action index
            if action in range(0, 5):
                obs, _, terminated, truncated, _ = env.step(action)

                log.write(f"Action: {env.unwrapped.action_type.ACTIONS_ALL[action]}\n\n")
                log.write(f"Response:\n{response}\n\n")

            else:
                # If the LLM gave an invalid response default to the IDLE action
                obs, _, terminated, truncated, _ = env.step(1)

                log.write(f"Fallback due to invalid action given by LLM.\nAction: IDLE")
                log.write(f"Response:\n{response}\n\n")
        else:
            # If LLM disabled, take IDLE action
            obs, _, terminated, truncated, _ = env.step(1)

        # If all timesteps of the episode have been executed or if the agent crashed then the episode is done
        # Terminated is true if the agent crashed, truncated is true if the max number of timesteps are reached
        done = terminated or truncated

        log.write("-------------------------------------------------\n\n")

    # After episode ends save the last frame
    # This is necessary to save the frame of a crash
    frame = env.render()
    Image.fromarray(frame).save(f"{episode_folder}/frame_{frame_count:05d}.png")

    timesteps.append(frame_count)
    episode_speeds.append(sum(speeds) / len(speeds))

    log.close()
    env.close()

episode_graph_folder = f"{output_folder}graphs"
os.makedirs(episode_graph_folder, exist_ok=True)

x = list(range(1, num_episodes + 1))
graph_plot(True, episode_graph_folder, "Average Speed", "Episode", "Speed (m/s)", x, episode_speeds, 30)
graph_plot(True, episode_graph_folder, "Timesteps Lasted", "Episode", "Timesteps", x, timesteps, 40)
graph_plot_point(True, episode_graph_folder, "Average Timesteps Lasted", "", "Timesteps", sum(timesteps) / len(timesteps), 40)
graph_plot_point(True, episode_graph_folder, "Average Success Rate", "", "Success Rate %", timesteps.count(40) / len(timesteps) * 100, 100)