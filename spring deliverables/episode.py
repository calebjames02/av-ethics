import gymnasium as gym
import os
from PIL import Image
from tools import Vehicle
from llm import ask_llm, prompt


def print_vehicles(vehicles):
    """
    Formats vehicle information grouped by lane into a readable string.

    Vehicles are separated into four lanes based on their `lane` attribute.
    For each lane, the function lists every vehicle along with its x position, x velocity, and y velocity. Empty lanes are labeled as "Empty".
    """

    # Create 4 empty lists, one for each lane
    lanes = [[] for _ in range(4)]
    output = ""

    # Store each vehicle in their corresponding lane
    for vehicle in vehicles:
        lanes[vehicle.lane - 1].append(vehicle)

    for i in range(len(lanes)):
        output += f"Lane: {i + 1}\n"

        if not len(lanes[i]):
            output += "Empty\n\n"
        else:
            for car in lanes[i]:
                output += f"{car.name}:\nx position = {float(car.x_pos)}, x velocity = {float(car.x_vel)}, y velocity = {float(car.y_vel)}\n\n"

        output += "\n"

    return output

def complete_episode(
        config: dict,
        model: str,
        episode_folder: str,
        use_llm: bool,
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

        # Create folder to save each frame of the environment to
        os.makedirs(episode_folder, exist_ok=True)

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
                log.write(print_vehicles(cars))

                if use_llm:
                    available = env.unwrapped.action_type.get_available_actions()
                    available_actions = [f"{action}: {env.unwrapped.action_type.ACTIONS_ALL[action]}" for action in available]

#                    print(env.unwrapped.action_type.ACTIONS_ALL)

                    # Action indeces don't match their position in the list which can be confusing for the LLM
                    # Sorting the list ensures that the index in the string matches its position in the list
                    available_actions.sort()

                    response, action = ask_llm(model, prompt, env.unwrapped.action_type.ACTIONS_ALL, cars)
#                    response, action = ask_llm(model, prompt, available_actions, cars)

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