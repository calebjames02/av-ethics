import gymnasium as gym
import highway_env
import os
import time
from graphs import graph_plot, graph_plot_point
from llm import ask_llm, prompt
from PIL import Image
from settings import load_settings, save_settings, DEFAULT_MODEL
from tools import closest_same_lane, Vehicle
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
from multiprocessing import Process, Array, Manager, Queue
from episode import complete_episode

NUM_WORKERS = 10

def process_func(
        q: multiprocessing.Queue,
        results: multiprocessing.Queue,
        config: dict,
        model: str,
        episode_folder: str,
        use_llm: bool,
        save_frames: bool,
    ) -> None:
        """
        Runs episodes and saves metrics while there is work to be done.        
        """
        while True:
            # Get episode number from passed in queue
            value = q.get(block=True)
            
            # No work left to do so terminate
            if value is None:
                break

            # Run episode and save metrics
            frame_count, speeds = complete_episode(config=config, model=model, episode_folder=episode_folder + f"_{value}", use_llm=use_llm, save_frames=save_frames)
            print(f"Episode {value} complete")
            results.put((value, frame_count, speeds))

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

        Creates and uses NUM_WORKERS sub processes to parallelize testing episodes
        Creates a timestamped directory for the test batch.
        Generates line and bar graphs tracking average speeds and survival rates.
        """
        processes = []

        # Make folder for episode output
        test_folder = f"{self.general_settings['output_folder']}/run_{self.run_count}_{int(time.time())}/"
        os.makedirs(test_folder, exist_ok=True)

        # Make a queue to store episode numbers, which the processes use to know when to work
        # multiprocessing.Queue() is used specifically because it can be safely accesses by multiple concurrent processes
        q = multiprocessing.Queue()
        for episode_num in range(1, episodes + 1):
            q.put(episode_num)

        # When processes get None instead of a number from the queue they know there is no work left to do and can terminate
        for _ in range(NUM_WORKERS):
            q.put(None)

        results = multiprocessing.Queue()
        episode_folder_base = f"{test_folder}{self.general_settings['output_subfolder']}_episode"
        model = next((k for k, v in self.llm_settings.items() if v and k != 'prompt'), DEFAULT_MODEL)
        print(model)

        for episode in range (NUM_WORKERS):
            args = (q, results, self.config, model, episode_folder_base, True, self.general_settings["save_frame_images"])

            # Create and start the process
            p = multiprocessing.Process(target=process_func, args=args)
            p.start()

            # Add the process to a list for access later
            processes.append(p)

        # Wait until each process is finished before proceeding
        for p in processes:
            p.join()

        values = []
        # Extract all return values from processes into a list
        while not results.empty():
            values.append(results.get())

        # If there are no return values then there is nothing to log, so just return
        if len(values) == 0:
            return

        # The values list is in the order episodes finish, so it needs to be sorted to ensure each episode shows up in order numerically
        values.sort()

        # Extract metrics from values list
        episodes, timesteps, speeds = map(list, zip(*values))
        speeds = [float(sum(s) / len(s)) if s else 0.0 for s in speeds]

        episode_graph_folder = test_folder + "graphs"
        os.makedirs(episode_graph_folder, exist_ok=True)

        graph_plot(self.graph_settings["episode_speed"], episode_graph_folder, "Average Speed", "Episode", "Speed (m/s)", episodes, speeds, 30)
        graph_plot(self.graph_settings["timesteps_lasted"], episode_graph_folder, "Timesteps Lasted", "Episode", "Timesteps", episodes, timesteps, 41)
        graph_plot_point(self.graph_settings["average_timesteps_lasted"], episode_graph_folder, "Average Timesteps Lasted", "", "Timesteps", sum(timesteps) / len(timesteps), 40)
        graph_plot_point(self.graph_settings["average_success_rate"], episode_graph_folder, "Average Success Rate", "", "Success Rate %", timesteps.count(40) / len(timesteps) * 100, 100)

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
                self.model = next((k for k, v in self.llm_settings.items() if v and k != 'prompt'), DEFAULT_MODEL)
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