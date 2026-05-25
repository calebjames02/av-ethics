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
import multiprocessing
from multiprocessing import Process, Array, Manager, Queue

def complete_episode(
        config: dict,
        use_llm: bool,
        episode_folder: str,
        model: str,
        save_frames: bool
    ) -> tuple[int, list[float]]:
        """
        Run a single simulation episode to completion (crash or timeout).
        
        Initializes the environment, captures frame data, formats the state space for 
        the LLM (if enabled), executes the chosen actions, and logs the process.

        Renders and saves individual frame images to disk (if enabled).
        Writes telemetry and LLM reasoning directly to a log.txt file.
        """

        env = gym.make('highway-v0', render_mode="rgb_array", config=config)

        state, _ = env.reset()
        frame_count = 0
        obs = state
        done = False
        speeds = []
        use_llm = False

        # Create folder to save each frame of the environment to
        os.makedirs(episode_folder, exist_ok=True)

        # Create file to log output to
#        log = open(f"{episode_folder}/log.txt", "w")

        # Set model to be whichever model is selected in settings
        # If somehow nothing is selected, then default to glm-4.7
#        self.model = next((k for k, v in self.llm_settings.items() if v and k != 'prompt'), "glm-4.7")
        print(model)

        with open(f"{episode_folder}/log.txt", "w") as log:
            while not done:
                # Render frame and save it locally if enabled
                frame = env.render()
                if save_frames:
    #            if self.general_settings["save_frame_images"]:
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

                    response, action = ask_llm(model, prompt, available_actions, cars)

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
        if save_frames:
#        if self.general_settings["save_frame_images"]:
            Image.fromarray(frame).save(f"{episode_folder}/frame_{frame_count:05d}.png")

        log.close()
        env.close()

        return frame_count, speeds