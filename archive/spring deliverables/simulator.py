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
    """
    Initialize the Simulator class.

    Loads settings from the configuration file, sets up tracking variables, 
    and defines the initial highway-env environment configuration.
    """

    def __init__(self):
        self.settings=load_settings()
        self.graph_settings = self.settings["graph_settings"]
        self.general_settings = self.settings["general_settings"]
        self.llm_settings = self.settings["llm_settings"]
        self.folder_name = self.general_settings["output_folder"]

        # Tracks number of times a test batch is executed, not individual episodes
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

    def save(self) -> None:
        """
        Save the current state of all settings back to the configuration file.
        
        This ensures any modifications made via the CLI are persisted for future runs.
        """
        save_settings(self.settings)

    def test(self, episodes: int) -> None:
        """
        Execute a batch of episodes and generate comprehensive telemetry outputs.

        Creates a timestamped directory for the test batch.
        Generates line and bar graphs tracking average speeds and survival rates.
        """
        self.timesteps, self.speeds = ([] for _ in range(2))

        # Make folder for episode output
        self.test_folder = f"{self.folder_name}/run_{self.run_count}_{int(time.time())}/"
        os.makedirs(self.test_folder, exist_ok=True)

        for episode in range (0, episodes):
            print(f"Episode: {episode}")
            self.episode_folder = f"{self.test_folder}{self.general_settings['output_subfolder']}_episode{episode + 1}_{int(time.time())}"
            frame_count, episode_speeds = self.complete_episode()
            self.speeds.append(sum(episode_speeds) / len(episode_speeds))
            self.timesteps.append(frame_count)

        # Create folder for graphs
        self.episode_graph_folder = self.test_folder + "graphs"
        os.makedirs(self.episode_graph_folder, exist_ok=True)

        x = list(range(1, episodes + 1))
        graph_plot(self.graph_settings["episode_speed"], self.episode_graph_folder, "Average Speed", "Episode", "Speed (m/s)", x, self.speeds, 30)
        graph_plot(self.graph_settings["timesteps_lasted"], self.episode_graph_folder, "Timesteps Lasted", "Episode", "Timesteps", x, self.timesteps, 40)
        graph_plot_point(self.graph_settings["average_timesteps_lasted"], self.episode_graph_folder, "Average Timesteps Lasted", "", "Timesteps", sum(self.timesteps) / len(self.timesteps), 40)
        graph_plot_point(self.graph_settings["average_success_rate"], self.episode_graph_folder, "Average Success Rate", "", "Success Rate %", self.timesteps.count(40) / len(self.timesteps) * 100, 100)

        self.run_count += 1

    def complete_episode(self) -> tuple[int, list[float]]:
        """
        Run a single simulation episode to completion (crash or timeout).
        
        Initializes the environment, captures frame data, formats the state space for 
        the LLM (if enabled), executes the chosen actions, and logs the process.

        Renders and saves individual frame images to disk (if enabled).
        Writes telemetry and LLM reasoning directly to a log.txt file.
        """

        self.env = gym.make('highway-v0', render_mode="rgb_array", config=self.config)

        state, _ = self.env.reset()
        frame_count = 0
        obs = state
        done = False
        speeds = []
        self.use_llm = False

        # Create folder to save each frame of the environment to
        os.makedirs(self.episode_folder, exist_ok=True)

        # Create file to log output to
        log = open(f"{self.episode_folder}/log.txt", "w")

        # Set model to be whichever model is selected in settings
        # If somehow nothing is selected, then default to glm-4.7
        self.model = next((k for k, v in self.llm_settings.items() if v and k != 'prompt'), "glm-4.7")
        print(self.model)

        while not done:
            # Render frame and save it locally if enabled
            frame = self.env.render()
            if self.general_settings["save_frame_images"]:
                Image.fromarray(frame).save(f"{self.episode_folder}/frame_{frame_count:05d}.png")

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
                available = self.env.unwrapped.action_type.get_available_actions()
                available_actions = [f"{action}: {self.env.unwrapped.action_type.ACTIONS_ALL[action]}" for action in available]

                # Action indeces don't match their position in the list which can be confusing for the LLM
                # Sorting the list ensures that the index in the string matches its position in the list
                available_actions.sort()

                response, action = ask_llm(self.model, prompt, available_actions, cars)

                # Check that LLM hasn't returned an invalid action index
                if action in range(0, 5):
                    obs, _, terminated, truncated, _ = self.env.step(action)

                    log.write(f"Action: {self.env.unwrapped.action_type.ACTIONS_ALL[action]}\n\n")
                    log.write(f"Response:\n{response}\n\n")

                else:
                    # If the LLM gave an invalid response default to the IDLE action
                    obs, _, terminated, truncated, _ = self.env.step(1)

                    log.write(f"Fallback due to invalid action given by LLM.\nAction: IDLE")
                    log.write(f"Response:\n{response}\n\n")
            else:
                # If LLM disabled, take IDLE action
                obs, _, terminated, truncated, _ = self.env.step(1)

            # If all timesteps of the episode have been executed or if the agent crashed then the episode is done
            # Terminated is true if the agent crashed, truncated is true if the max number of timesteps are reached
            done = terminated or truncated

            log.write("-------------------------------------------------\n\n")

        # After episode ends save the last frame
        # This is necessary to save the frame of a crash
        frame = self.env.render()
        if self.general_settings["save_frame_images"]:
            Image.fromarray(frame).save(f"{self.episode_folder}/frame_{frame_count:05d}.png")

        log.close()
        self.env.close()

        return frame_count, speeds

    def modify_settings(self, setting_val: str) -> None:
        """
        Launch an interactive CLI menu for the user to modify a specific category of settings.
        """

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
                index = int(index)

                if(index < 0 or index > len(setting_subtype)):
                    print("Number entered is out of valid range\n")

                else:
                    if index == 0:
                        return

                    # The options are printed as one indexed, but are stored as zero indexed
                    # Need to subtract 1 to make the index zero indexed
                    index -= 1

                    print()

                    setting_key = list(setting_subtype.keys())[index]
                    setting_val = setting_subtype[setting_key]

                    # Use a multiple-choice menu for boolean settings and text input for everything else.
                    # LLM settings are handled exclusively to swap the active model rather than just toggling it.
                    if type(setting_val) is bool:
                        if is_llm_setting:
                            if not setting_val:
                                self.modify_item_set([setting_key, setting_val], self.llm_settings, {"1": True})

                                # Set previously enabled model to false
                                self.llm_settings[self.model] = False

                            else:
                                print(f"Model {self.model} is already selected, skipping.\n")

                        else:
                            self.modify_item_set([setting_key, setting_val], setting_subtype, {"1": True, "2": False})

                    else:
                        self.modify_item([setting_key, setting_val], setting_subtype)

            # Display error message in case of user giving non-numerical input
            except:
                print("Textual input is not valid\n")

            # Wait one second before looping again to give time for people to read any program output
            time.sleep(1)

    def modify_item(self,
            setting: list,
            setting_class: dict
        ) -> None:
        """
        Allows the user to input text to modify a specific setting via the CLI.
        """

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

        setting_class[setting[0]] = val

    def modify_item_set(self,
            setting: tuple[str, str],
            setting_class: dict,
            options: dict
        ) -> None:
        """
        Allows the user to choose from certain options to modify a specific setting via the CLI.
        """

        print(f"{setting[0]} is currently {setting[1]}")
        while 1:
            print("What would you like to change it to? ", end="")
            for key in options:
                print(f"{key}: {options[key]} ", end="")
            val = input("\nYour response: ")

            print(f"\nYou entered: '{options[val]}'")
            done = input("Is this correct? Yes: 1 | No: 2 | Cancel: 3\nYour response: ")
            print()

            # If user input is 1, break from loop and save setting
            if done == "1":
                break
            # If user input is 3, return from function and don't save setting
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