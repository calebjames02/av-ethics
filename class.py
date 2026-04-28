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
from openai import RateLimitError

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
#CRITICAL: You MUST respond ONLY in raw JSON. 
#Do NOT wrap your response in markdown code blocks (e.g., ```json). 
#Do NOT output any conversational text. Your entire response must start with { and end with }."""

def ask_llm(model, prompt, actions, state, closest):
    # Instead of beta.parse, we use the instructor-patched create method.
    # This automatically handles prompt injection, schema validation, and retry loops.
    try:
        decision_obj = client.chat.completions.create(
            model=model,
            response_model=DrivingDecision,
            max_retries=3,
            temperature=0.0, # CRITICAL FIX: Forces deterministic output and prevents the infinite whitespace/newline loop
            max_tokens=1000, # Fail-safe: Prevents the model from hanging for 3 minutes if it does start looping
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"State space: {state}"},
                {"role": "user", "content": f"Available actions: {actions}"},
            ],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )

        return decision_obj.explanation, decision_obj.selected_action_index

    except RateLimitError as e:
        print("⚠️ Rate limited!")
        print(e)

        # You can implement backoff/retry here
        return "Rate limited. Try again later.", 1

    except Exception as e:
        print(f"API Error or Parsing Failure: {e}")
        # Return fallback safe values so the simulation doesn't crash entirely on API failure
        return "Fallback due to API parsing error.", 1

"""
prompt = "You are an intelligent driving assistant whose goal is to drive safely and efficiently. You are directing the ego vehicle in this simulation, and your job is to select the best action given a list of possible actions and the state space at a specific time step. Explain your reasoning thoroughly for each candidate action before selecting the most optimal action."
def ask_llm(model, prompt, actions, state, closest):
    completion = client.chat.completions.create(
#    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            # These additional messages give Chat GPT more context on how to interpret what is given to it
            {"role": "user", "content": f"State space: {state}"},
            {"role": "user", "content": f"Available actions: {actions}"},
#            {"role": "user", "content": f"{closest}"},
        ],
        response_model=DrivingDecision,
    )

#    decision_obj = completion.choices[0].message.parsed

    return completion.explanation, completion.selected_action_index
"""

"""
Purpose: From the cars list find the closest vehicle in front of the ego vehicle, if one exists
"""
def closest_same_lane(cars):
    if not cars:
        return "No cars currently exist"

    if len(cars) == 1:
        return "Only the ego vehicle exists"

    # Mark down ego lane and position for future access
    ego_lane = cars[0].lane
    ego_pos = cars[0].x_pos

    # Filter vehicles out of lane_cars list that aren't in the same lane as the ego vehicle
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
    # Save current version of settings back to SETTINGS_FILE
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)

def graph_plot(enabled, folder_path, graph_title, x_data, y_data, x_label, y_label, y_max):
    # Don't save graph if it isn't enabled
    if not enabled:
        return

    # If there is only one data point make the graph a bar graph instead of a line graph
    if len(x_data) == 1:
        graph_plot_point(enabled, folder_path, graph_title, 0, y_data[0], "", y_label, y_max)
        return

    # Graph setup
    fig, ax = plt.subplots()
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

def graph_plot_point(enabled, folder_path, graph_title, x_data, y_data, x_label, y_label, y_max):
    # Don't save graph if it isn't enabled
    if not enabled:
        return

    # Graph setup
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(graph_title)

    # Plot data
    ax.bar([x_data], max(0.005, y_data))
    
    # Setup x and y limits, tick marks
    ax.set_xlim(-1e-12, 1e-12)
    ax.set_ylim(max(0, y_data - 1), min(y_max, y_data + 1))
    ax.set_xticks(range(0, 0, 1))

    # Save image locally
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
        self.llm_models = self.llm_settings.copy()
        self.llm_models.pop('prompt')
        self.folder_name = self.general_settings["output_folder"]
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
        self.timesteps, self.speeds = ([] for _ in range(2))
        self.test_folder = f"{self.folder_name}/run_{self.run_count}_{int(time.time())}/"
        os.makedirs(self.test_folder, exist_ok=True)
        for episode in range (0, episodes):
            print(f"Episode: {episode}")
            self.episode_folder = f"{self.test_folder}{self.general_settings['output_subfolder']}_episode{episode + 1}_{int(time.time())}"
            frame_count, episode_speeds = self.complete_episode()
            self.speeds.append(sum(episode_speeds) / len(episode_speeds))
            self.timesteps.append(frame_count)

        x = list(range(episodes))
        self.episode_graph_folder = self.test_folder + "graphs"
        os.makedirs(self.episode_graph_folder, exist_ok=True)
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
        self.model = next((k for k, v in self.llm_models.items() if v), "glm-4.7")
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
                # If LLM enabled, prompt LLM with list of cars and background prompt then take the specified action
                print("Started processing")
                start = time.time()
                response, action = ask_llm(self.model, prompt, ACTIONS_ALL, cars, "")
                end = time.time()
                print(f"Processing took {end-start} seconds")
                if check_valid_action(action):
                    # Take given action in environment
                    next_state, _, terminated, truncated, _ = self.env.step(action)

                    # Log action and response
                    log.write(f"Action: {ACTIONS_ALL_DICT[action]}\n\n")
                    log.write(f"Response:\n{response}\n\n")

                else:
                    next_state, _, terminated, truncated, _ = self.env.step(1)
                    log.write(f"Fallback due to invalid action given by LLM. Action: IDLE")
                    log.write(f"Response:\n{response}\n\n")
            else:
                # If LLM disabled, take IDLE action
                print(check_valid_action(1))
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

    def modify_llm_settings(self):
        while 1:
            # Set model to be whichever model is selected in settings
            # If somehow nothing is selected, then default to glm-4.7

            self.model = next((k for k, v in self.llm_models.items() if v), "glm-4.7")
            print(f"Current model: {self.model}")
            settings_list = {}
            count = 1
            for key, value in self.llm_settings.items():
                settings_list[count] = (key, value)
                count += 1

            print("0: Exit")
            for key, value in settings_list.items():
                print(f"{key}: {value[0]} = {value[1]}")
            print()

            val = input("Which setting would you like to modify?: ")

            try:
                val = int(val)
                if(val < 0 or val > len(settings_list)):
                    print("Number entered is out of valid range\n")
                else:
                    if val == 0:
                        return

                    print()
                    if type(settings_list[val][1]) is bool:
                        if not settings_list[val][1]:
                            self.llm_settings[self.model] = False
                            self.modify_item_set(settings_list[val], self.llm_settings, {"1": True})
                        else:
                            print(f"Model {self.model} is already selected, skipping.\n")

                    else:
                        self.modify_item(settings_list[val], self.llm_settings)
            except:
                print("Textual input is not valid\n")

            # Wait one second before looping again to give time for people to read any program output
            time.sleep(1)

    def modify_settings(self, setting_val):
        setting_subtype = self.settings[setting_val]
        while 1:
            print("Current settings:")
            settings_list = {}
            count = 1

            for key, value in setting_subtype.items():
                settings_list[count] = (key, value)
                count += 1

            print("0: Exit")
            for key, value in settings_list.items():
                print(f"{key}: {value[0]} = {value[1]}")
            print()

            val = input("Which setting would you like to modify?: ")

            try:
                val = int(val)
                if(val < 0 or val > len(settings_list)):
                    print("Number entered is out of valid range\n")
                else:
                    if val == 0:
                        return
                    print()
                    if type(settings_list[val][1]) is bool:
                        self.modify_item_set(settings_list[val], setting_subtype, {"1": True, "2": False})
                    else:
                        self.modify_item(settings_list[val], setting_subtype)
            except:
                print("Textual input is not valid\n")

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
            sim.modify_llm_settings()
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