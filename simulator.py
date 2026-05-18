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

class Simulator():
    def __init__(self):
        self.settings=load_settings()
        self.graph_settings = self.settings["graph_settings"]
        self.general_settings = self.settings["general_settings"]
        self.llm_settings = self.settings["llm_settings"]
        self.folder_name = self.general_settings["output_folder"]

        # Used to keep track of how many test runs are completed
        # Doesn't track number of episodes run, instead tracks number of time 1 or more episodes are run
        self.run_count = 1

        self.config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": True,
                "normalize": False
            }
        }

    def save(self):
        save_settings(self.settings)

    def test(self, episodes):
        # Create empty lists to track number of timesteps and average speed from each episode
        self.timesteps, self.speeds = ([] for _ in range(2))

        # Make folder for episode output
        self.test_folder = f"{self.folder_name}/run_{self.run_count}_{int(time.time())}/"
        os.makedirs(self.test_folder, exist_ok=True)

        # Run each episode
        for episode in range (0, episodes):
            print(f"Episode: {episode}")
            self.episode_folder = f"{self.test_folder}{self.general_settings['output_subfolder']}_episode{episode + 1}_{int(time.time())}"
            frame_count, episode_speeds = self.complete_episode()
            self.speeds.append(sum(episode_speeds) / len(episode_speeds))
            self.timesteps.append(frame_count)

        # Create folder for graphs
        self.episode_graph_folder = self.test_folder + "graphs"
        os.makedirs(self.episode_graph_folder, exist_ok=True)

        # Save all graphs
        x = list(range(episodes))
        graph_plot(self.graph_settings["episode_speed"], self.episode_graph_folder, "Average Speed", x, self.speeds, "Episode", "Speed (m/s)", 25)
        graph_plot(self.graph_settings["timesteps_lasted"], self.episode_graph_folder, "Timesteps Lasted", x, self.timesteps, "Episode", "Timesteps", 40)
        graph_plot_point(self.graph_settings["average_timesteps_lasted"], self.episode_graph_folder, "Average Timesteps Lasted", 0, sum(self.timesteps) / len(self.timesteps), "", "Timesteps", 40, )
        graph_plot_point(self.graph_settings["average_success_rate"], self.episode_graph_folder, "Average Success Rate", 0, self.timesteps.count(40) / len(self.timesteps) * 100, "", "Success Rate %", 100)

        self.run_count += 1

    def complete_episode(self):
        # Create environment
        self.env = gym.make('highway-v0', render_mode="rgb_array", config=self.config)

        # Variable initialization
        state, _ = self.env.reset()
        frame_count = 0
        obs = state
        done = False
        speeds = []

        # Used to control whether or not LLM is used
        self.use_llm = True

        # Create folder to save each frame of the environment to
        os.makedirs(self.episode_folder, exist_ok=True)

        # Create file to log output to
        log = open(f"{self.episode_folder}/log.txt", "w")

        # Set model to be whichever model is selected in settings
        # If somehow nothing is selected, then default to glm-4.7
        self.model = next((k for k, v in self.llm_settings.items() if v and k != 'prompt'), "glm-4.7")
        print(self.model)

        while not done:
            # Save frame to frame list
            frame = self.env.render()
            if self.general_settings["save_frame_images"]:
                Image.fromarray(frame).save(f"{self.episode_folder}/frame_{frame_count:05d}.png")

            # Make list of all cars currently on the road
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

            if self.use_llm:
                # Generate list of legal actions and their associated indices using the environment get_available_actions function
                available = self.env.unwrapped.action_type.get_available_actions()
                print(frame_count)
#                print(available)
                available_actions = [f"{action}: {self.env.unwrapped.action_type.ACTIONS_ALL[action]}" for action in available]
                available_actions.sort()

#                print(available_actions)
                print()
                # If LLM enabled, prompt LLM with list of cars and background prompt then take the specified action
                response, action = ask_llm(self.model, prompt, available_actions, cars, "")

                # Check that LLM hasn't returned an invalid action index
                if action in range(0, 5):
                    # Take given action in environment
                    next_state, _, terminated, truncated, _ = self.env.step(action)

                    # Log action and response
                    log.write(f"Action: {self.env.unwrapped.action_type.ACTIONS_ALL[action]}\n\n")
                    log.write(f"Response:\n{response}\n\n")

                else:
                    # Take given action in environment
                    next_state, _, terminated, truncated, _ = self.env.step(1)

                    # Log action and response
                    log.write(f"Fallback due to invalid action given by LLM.\nAction: IDLE")
                    log.write(f"Response:\n{response}\n\n")
            else:
                print(self.env.unwrapped.action_type.ACTIONS_ALL)
                # If LLM disabled, take IDLE action
                next_state, _, terminated, truncated, _ = self.env.step(1)

            # Update observations to be the new state
            obs = next_state

            # If all timesteps of the episode have been executed or if the agent crashed then the episode is done
            # Terminated is true if the agent crashed, truncated is true if the max number of timesteps are reached
            done = terminated or truncated

            log.write("-------------------------------------------------\n\n")

        # After episode ends save the last frame
        # This is necessary to save the frame of a crash
        frame = self.env.render()
        if self.general_settings["save_frame_images"]:
            Image.fromarray(frame).save(f"{self.episode_folder}/frame_{frame_count:05d}.png")

        # Close log file and environment
        log.close()
        self.env.close()

        return frame_count, speeds

    def modify_settings(self, setting_val):
        setting_subtype = self.settings[setting_val]

        # Check if llm settings are being modified, as certain extra things need to be done if this is true
        is_llm_setting = setting_val == "llm_settings"
        while 1:
            if is_llm_setting:
                # Set model to be whichever model is selected in settings
                # If somehow nothing is selected, then default to glm-4.7
                self.model = next((k for k, v in self.llm_settings.items() if v and k != 'prompt'), "glm-4.7")
                print(f"Current model: {self.model}")

            # Print all settings and their values
            print("0: Exit")
            for index, (key, value) in enumerate(setting_subtype.items(), start=1):
                print(f"{index}: {key} = {value}")
            print()

            index = input("Which setting would you like to modify?: ")

            try:
                # Convert index to string
                index = int(index)

                # Ensure index is in valid range
                if(index < 0 or index > len(setting_subtype)):
                    print("Number entered is out of valid range\n")

                else:
                    # If user input is 0, break from loop
                    if index == 0:
                        return

                    # Make index zero indexed
                    # It's originally not zero indexed to match up with the number of 0 being exit
                    index -= 1

                    print()

                    setting_key = list(setting_subtype.keys())[index]
                    setting_val = setting_subtype[setting_key]

                    # Handle modifying boolean setting
                    if type(setting_val) is bool:
                        # If llm setting is being modified handle it specifically
                        if is_llm_setting:
                            # If llm model is not enabled then give user option to enable it
                            if not setting_val:
                                # Set previously enabled model to false
                                self.llm_settings[self.model] = False

                                self.modify_item_set([setting_key, setting_val], self.llm_settings, {"1": True})
                            # If llm model is already enabled then don't change anything
                            else:
                                print(f"Model {self.model} is already selected, skipping.\n")

                        # Update any non llm setting normally
                        else:
                            self.modify_item_set([setting_key, setting_val], setting_subtype, {"1": True, "2": False})

                    # Handle modifying non boolean setting
                    else:
                        self.modify_item([setting_key, setting_val], setting_subtype)

            # Display error message in case of user giving non-numerical input
            except:
                print("Textual input is not valid\n")

            # Wait one second before looping again to give time for people to read any program output
            time.sleep(1)

    def modify_item(self, setting, setting_class):
        print(f"{setting[0]} is currently: {setting[1]}")
        while 1:
            val = input("What would you like to change it to?: ")
            print()
            print(f"You entered: '{val}'")
            done = input("Is this correct? Yes: 1 No: 2 Cancel: 3\nResponse: ")
            print()

            # If user input is 1, break from loop and save setting
            if done == "1":
                break
            # If user input is 3, return from function and don't save setting
            elif done == "3":
                return
            # If user input is 2, the loop continues

        setting_class[setting[0]] = val

    def modify_item_set(self, setting, setting_class, options):
        print(f"{setting[0]} is currently {setting[1]}")
        while 1:
            print("What would you like to change it to? ", end="")
            for key in options:
                print(f"{key}: {options[key]} ", end="")
            val = input("\nYour response: ")

            print()
            print(f"You entered: '{options[val]}'")
            done = input("Is this correct? Yes: 1 | No: 2 | Cancel: 3\nYour response: ")
            print()
            if done == "1":
                break
            elif done == "3":
                return

        setting_class[setting[0]] = options[val]

sim = Simulator()

while(1):
    val = input("0: Exit\n1: General settings\n2: LLM Settings\n3: Graphs\n4: Test agent\nYour choice: ")
    print()
    
    match val:
        case "0":
            break
        case "1":
            sim.modify_settings("general_settings")
        case "2":
            sim.modify_settings("llm_settings")
        case "3":
            sim.modify_settings("graph_settings")
        case "4":
            while True:
                val = input("How many episodes do you want run? (Type '0' to go back): ")

                if val == '0':
                    break

                try: 
                    val = int(val)
                    sim.test(val)
                except:
                    print("Invalid input\n")

sim.save()