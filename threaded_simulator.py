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
from episode import complete_episode

NUM_WORKERS = 20

def process_func(
        config: dict,
        use_llm: bool,
        episode_folder: str,
        model: str,
        save_frames: bool,
        q: multiprocessing.Queue,
        return_values: multiprocessing.Queue
    ) -> None:
        while True:
            value = q.get(block=True)

            if value is None:
                break

            frame_count, speeds = complete_episode(config=config, use_llm=use_llm, episode_folder=episode_folder + f"_{value}", model=model, save_frames=save_frames)
            print(f"{value}: {frame_count}")
            return_values.put((value, frame_count, speeds))

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
        processes = []

        # Make folder for episode output
        self.test_folder = f"{self.folder_name}/run_{self.run_count}_{int(time.time())}/"
        os.makedirs(self.test_folder, exist_ok=True)
        q = multiprocessing.Queue()
        return_values = multiprocessing.Queue()

        for episode_num in range(episodes):
            q.put(episode_num)

        for _ in range(NUM_WORKERS):
            q.put(None)

        episode_folder_base = f"{self.test_folder}{self.general_settings['output_subfolder']}_episode"
        model = next((k for k, v in self.llm_settings.items() if v and k != 'prompt'), "glm-4.7")

        for episode in range (NUM_WORKERS):
            print(f"Episode: {episode}")
#            self.episode_folder = f"{self.test_folder}{self.general_settings['output_subfolder']}_episode{episode + 1}_{int(time.time())}"
            args = (self.config, False, episode_folder_base, model, self.general_settings["save_frame_images"], q, return_values)
            #frame_count, episode_speeds = self.complete_episode(config=self.config, use_llm=False, episode_folder=self.episode_folder, model=self.model, save_frames=self.general_settings["save_frame_images"])
            p = multiprocessing.Process(target=process_func, args=args)
            p.start()
            processes.append(p)
#            p.join()
#            self.speeds.append(sum(episode_speeds) / len(episode_speeds))
#            self.timesteps.append(frame_count)

        for p in processes:
            p.join()

#        values = []
#        while not return_values.empty():
#            values.append(return_values.get())
        values = []
        for _ in range(episodes):
            try:
                values.append(return_values.get(timeout=30))
            except Exception:
                pass
        if len(values) == 0:
            return
        values.sort()
#        print([v for v, _, _ in values])
        episodes, self.timesteps, self.speeds = map(list, zip(*values))
        #self.timesteps = [v for _, v, _ in values]
        print(self.timesteps)
        #self.speeds = [v for _, _, v in values]
        self.speeds = [float(sum(s) / len(s)) if s else 0.0 for s in self.speeds]
        print(self.speeds)

        # Create folder for graphs
        self.episode_graph_folder = self.test_folder + "graphs"
        os.makedirs(self.episode_graph_folder, exist_ok=True)

        x = [v for v, _, _ in values]
        graph_plot(self.graph_settings["episode_speed"], self.episode_graph_folder, "Average Speed", "Episode", "Speed (m/s)", x, self.speeds, 25)
        graph_plot(self.graph_settings["timesteps_lasted"], self.episode_graph_folder, "Timesteps Lasted", "Episode", "Timesteps", x, self.timesteps, 40)
        graph_plot_point(self.graph_settings["average_timesteps_lasted"], self.episode_graph_folder, "Average Timesteps Lasted", "", "Timesteps", sum(self.timesteps) / len(self.timesteps), 40)
        graph_plot_point(self.graph_settings["average_success_rate"], self.episode_graph_folder, "Average Success Rate", "", "Success Rate %", self.timesteps.count(40) / len(self.timesteps) * 100, 100)

        self.run_count += 1

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

#                try: 
                val = int(val)
                sim.test(val)
#                except:
#                    print("Invalid input\n")

sim.save()