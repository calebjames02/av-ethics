import gymnasium as gym
import highway_env
import os
import json
import time
import instructor
from agents import Agent, SQLiteSession
from dataclasses import dataclass
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class Vehicle:
    name: str
    x_pos: float
    lane: int
    x_vel: float
    y_vel: float

# Access OpenAI API key from key.txt file
with open("nautilus_key.txt") as file:
    key = file.readline().strip()
os.environ["OPENAI_API_KEY"] = key
#client = instructor.from_openai(OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://ellm.nrp-nautilus.io/v1"))
client = instructor.from_openai(
    OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://ellm.nrp-nautilus.io/v1"),
    mode=instructor.Mode.MD_JSON # Forces JSON mode + Prompt Injection across compatible APIs
)

# Discrete action space
ACTIONS_ALL = [
    "0: LANE_LEFT",
    "1: IDLE",
    "2: LANE_RIGHT",
    "3: FASTER",
    "4: SLOWER"
]
ACTIONS_ALL_DICT = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER"

}

# Structures the response into a separate explanation and action
# This ensures the action can be properly identified even if the format of the response changes
class DrivingDecision(BaseModel):
    explanation: str
    selected_action_index: int

prompt = """You are an intelligent driving assistant whose goal is to drive safely and efficiently. 
You are directing the ego vehicle in this simulation, and your job is to select the best action given a list of possible actions and the state space at a specific time step. 
Explain your reasoning thoroughly for each candidate action before selecting the most optimal action."""

def ask_llm(model, prompt, actions, state, closest):
    try:
        decision_obj = client.chat.completions.create(
            model=model,
            response_model=DrivingDecision,
            max_retries=1,
            temperature=0.0, # Set temperature to 0.0 to force deterministic output
            max_tokens=1000, # Stop llm from infinitely outputing text
            timeout=20, # Automatic timeout if processing too slow
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"State space: {state}"},
                {"role": "user", "content": f"Available actions: {actions}"},
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )

        return decision_obj.explanation, decision_obj.selected_action_index

    except Exception as e:
        print(f"API Error or Parsing Failure: {e}")
        # Return fallback safe values so the simulation doesn't crash entirely on API failure
        return "Fallback due to API parsing error.", 1

"""
Purpose: From the cars list find the closest vehicle in front of the ego vehicle, if one exists
Input: List of cars on the road
Output: Textual description of position of vehicle closest to ego vehicle
"""
def closest_same_lane(cars):
    if not cars:
        return "No cars currently exist"

    if len(cars) == 1:
        return "Only the ego vehicle exists"

    # Mark down ego lane and position for future access
    ego_lane = cars[0].lane
    ego_pos = cars[0].x_pos

    # Filter vehicles out of cars list that aren't in the same lane as the ego vehicle
    lane_cars = [car for car in cars if car.lane == ego_lane]

    if(len(lane_cars) == 1):
        return f"There is no car close to the ego vehicle in lane {ego_lane}"

    closest_car = None
    min_dist = float('inf')
        
    for car in lane_cars[1:]:
        dist = car.x_pos - ego_pos
        if dist > 0 and dist < min_dist:
            min_dist = dist
            closest_car = car

    if min_dist == float('inf'):
        return f"There is no car close to the ego vehicle in lane {ego_lane}"

    return f"The closest vehicle to the ego vehicle in lane {ego_lane} is {closest_car.name} at position x = {closest_car.x_pos}"

SETTINGS_FILE = "settings.json"

DEFAULT_SETTINGS = {
    "general_settings": {
    "tensorboard_writer": "runs",
    "output_folder": "frames",
    "output_subfolder": "run",
    "save_frame_images": True,
    }, 
    "graph_settings": {
        "average_success_rate": True,
        "average_timesteps_lasted": True,
        "episode_speed": True,
        "timesteps_lasted": True,
    },
    "llm_settings": {
        "qwen3": False,
        "gpt-oss": True,
        "glm-4.7": False,
        "gemma": False,
        "olmo": False,
        "minimax-m2": False,
        "prompt": "You are an intelligent driving assistant whose goal is to drive safely and efficiently. You are directing the ego vehicle in this simulation, and your job is to select the best action given a list of possible actions and the state space at a specific time step. Explain your reasoning thoroughly for each candidate action before selecting the most optimal action.",
    }
}

"""
Purpose: Extract current copy of settings from SETTINGS_FILE if it exists, otherwise make a copy of the default settings and return that
Output: Dictionary containing copy of saved settings, or default settings
"""
def load_settings():
    # If SETTINGS_FILE already exists, extract current settings from it
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            current = json.load(f)
            
            # Add any values that are in the default_settings and not the json file
            for key, value in DEFAULT_SETTINGS.items():
                # If value type is a dict, expand it and check that all keys of the dictionary are present
                if type(value) is dict:
                    for k2, v2 in DEFAULT_SETTINGS[key].items():
                        if k2 not in current[key]:
                            current[key][k2] = v2
                # If value is of a scalar type, update it if missing
                if key not in current:
                    current[key] = value

            return current

    # If SETTINGS_FILE doesn't exist, return DEFAULT_SETTINGS
    else:
        return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    """
    Save current version of settings to SETTINGS_FILE
    Should be called whenver program is terminated to save any changes that were made

    Args:
        settings (dict):
            Dictionary that contains the current state of all the settings

    Returns:
        None
    """

    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)

def graph_plot(enabled, folder_path, graph_title, x_data, y_data, x_label, y_label, y_max):
    """
    Create and save a line graph display the passed in data.
    The chart is saved as a PNG image in the specified folder.

    Args:
        enabled (bool):
            Whether graph generation is enabled.

        folder_path (str):
            Directory where the graph image will be saved.

        graph_title (str):
            Title of the graph and filename of the output image.

        x_data (float):
            .

        y_data (float):
            .

        x_label (str):
            Label for the x-axis.

        y_label (str):
            Label for the y-axis.

        y_max (float):
            Maximum allowed y-axis limit.

    Returns:
        None

    Notes:
    """

    # The enabled parameter tracks whether or not the given graph is enabled in settings
    # If it is not enabled, don't graph it
    if not enabled:
        return

    # If there is only one data point make the graph a bar graph instead of a line graph
    if len(x_data) == 1:
        graph_plot_point(enabled, folder_path, graph_title, 0, y_data[0], "", y_label, y_max)
        return

    # Graph setup
    _, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(graph_title)

    # Plot data
    ax.plot(list(range(1, len(y_data) + 1)), y_data)

    # Set limits for y values
    ax.set_ylim(max(0, min(y_data) - 1), min(y_max + 0.025, max(y_data) + 1))

    # Ensure tick marks on x axis are only ever whole numbers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Save graph locally
    plt.savefig(f"{folder_path}/{graph_title}.png")
    plt.close()

"""
Purpose: Creates and saves a bar chart containing a single bar
Input: Enabled - boolean value representing whether or not the graph should be made
    folder_path - string containing the file path of the folder to save the graph to
    graph_title - string containing a title for the graph
    x_data / y_data - numeric data to graph on given axis
    x_label / y_label - string containing name for given axis
    y_max - maximum value for y bars
Output: None
Notes: A bar graph is used instead of a line graph to make it clearer what the value is
"""
def graph_plot_point(enabled, folder_path, graph_title, x_value, y_value, x_label, y_label, y_max):
    """
    Create and save a bar chart containing a single data point.
    The chart is saved as a PNG image in the specified folder.

    Args:
        enabled (bool):
            Whether graph generation is enabled.

        folder_path (str):
            Directory where the graph image will be saved.

        graph_title (str):
            Title of the graph and filename of the output image.

        x_value (float):
            X-axis value for the bar position.

        y_value (float):
            Height of the bar.

        x_label (str):
            Label for the x-axis.

        y_label (str):
            Label for the y-axis.

        y_max (float):
            Maximum allowed y-axis limit.

    Returns:
        None

    Notes:
        A minimum bar height of 0.005 is enforced to ensure visibility for very small values.
        A bar graph is used instead of a line graph to make it clearer what the value is
    """

    # The enabled parameter tracks whether or not the given graph is enabled in settings
    # If it is not enabled, don't graph it
    if not enabled:
        return

    # Graph setup
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(graph_title)

    # Plot data
    ax.bar([x_value], max(0.005, y_value))
    
    # Setup x and y limits, tick marks
    ax.set_xlim(-1e-12, 1e-12)

    # Disables x ticks
    ax.set_xticks([])
    
    # Keep y limits close to y value, but make sure it is within the range (0, y_max)
    ax.set_ylim(max(0, y_value - 1), min(y_max, y_value + 1))

    # Save image locally in given folder with given title
    plt.tight_layout()
    plt.savefig(f"{folder_path}/{graph_title}.png")
    plt.close()

def check_valid_action(action):
    return action in range (0, 5)

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