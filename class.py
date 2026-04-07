import gymnasium as gym
import highway_env
import os
import json
import time
from agents import Agent, SQLiteSession
from dataclasses import dataclass
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel
from torch.utils.tensorboard import SummaryWriter
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
with open("key.txt") as file:
    key = file.readline().strip()
os.environ["OPENAI_API_KEY"] = key
client = OpenAI()

# Discrete action space
ACTIONS_ALL = [
    "0: LANE_LEFT",
    "1: IDLE",
    "2: LANE_RIGHT",
    "3: FASTER",
    "4: SLOWER"
]

def return_action(action):
    if action == 0:
        return "LANE_LEFT"
    
    if action == 1:
        return "IDLE"
    
    if action == 2:
        return "LANE_RIGHT"
    
    if action == 3:
        return "FASTER"
    
    if action == 4:
        return "SLOWER"

# Structures the response into a separate explanation and action
# This ensures the action can be properly identified even if the format of the response changes
class DrivingDecision(BaseModel):
    explanation: str
    selected_action_index: int

agent = Agent(
	name="Assistant",
	instructions="You are an intelligent driving assistant whose goal is to drive safely and efficiently. You are directing the ego vehicle in this simulation, and your job is to select the best action given a list of possible actions and the state space at a specific time step. Explain your reasoning for each action you choose.",
)

session = SQLiteSession("conversation_123")

prompt = "You are an intelligent driving assistant whose goal is to drive safely and efficiently. You are directing the ego vehicle in this simulation, and your job is to select the best action given a list of possible actions and the state space at a specific time step. Explain your reasoning thoroughly for each candidate action before selecting the most optimal action."
def ask_chat_gpt(prompt, actions, state, closest):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",

        messages=[
            {"role": "system", "content": prompt},
            # These additional messages give Chat GPT more context on how to interpret what is given to it
            {"role": "user", "content": f"State space: {state}"},
            {"role": "user", "content": f"Available actions: {actions}"},
#            {"role": "user", "content": f"{closest}"},
        ],
        response_format=DrivingDecision,
    )

    decision_obj = completion.choices[0].message.parsed

    return decision_obj.explanation, decision_obj.selected_action_index

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
    "tensorboard_writer": "experiment_test",
    "output_folder": "frames_test",
    "output_subfolder": "run_test",
    "save_frame_images": "true"
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            current = json.load(f)
            
            # Add any values that are in the default_settings and not the json file
            for key, value in DEFAULT_SETTINGS.items():
                if key not in current:
                    current[key] = value

            return current
    else:
        return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)

def graph_plots(graph_title, x_data, y_data, x_label, y_label, x_start, x_end, x_amt, y_start, y_end, y_amt):
    fig, ax = plt.subplots()
    ax.plot(x_data, y_data)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(graph_title)

    ax.set_yticks(range(y_start, y_end, y_amt))

    # Enable tick marks
    ax.tick_params(axis='both', which='both', length=6)

    ax.set_ylim(min(y_data), max(y_data))
    ax.set_xticks(range(x_start, x_end, x_amt))

    plt.savefig(f"{graph_title}.png")

def graph_plot_point(x_data, y_data, x_label, y_label, graph_title):
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.bar([x_data], [y_data])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(graph_title)

    ax.set_xlim(x_data, x_data)
    ax.set_xticks(range(0, 0, 1))
    ax.set_ylim(y_data - 1, y_data + 1)

    plt.tight_layout()
    plt.savefig(f"{graph_title}.png")

class Simulator():
    def __init__(self):
        self.settings=load_settings()
        self.folder_name = self.settings["output_folder"]
        self.run_count = 1
        self.time = time.time()

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
        self.crashed = []
        self.timesteps = []
        self.speeds = []
        self.writer = SummaryWriter(f"{self.settings['tensorboard_writer']}/{self.time} - test {self.run_count}")
        self.run_count += 1
        for episode in range (0, episodes):
            frame_count, episode_speeds = self.complete_episode()
            self.speeds.append(sum(episode_speeds) / len(episode_speeds))
            self.timesteps.append(frame_count)
            self.crashed.append(0 if frame_count == 40 else 1)

            self.writer.add_scalar(f"Timesteps lasted", frame_count, episode)
            self.writer.add_scalar(f"Average Speed", sum(episode_speeds) / len(episode_speeds), episode)
            self.writer.flush()

        x = list(range(episodes))
        graph_plots("Episode Speed", x, self.speeds, "Episode", "Speed (m/s)", 0, episodes, 1, 20, 26, 1)
        graph_plots("Timesteps Lasted", x, self.timesteps, "Episode", "Timesteps", 0, episodes, 1, 0, 40, 2)
        graph_plot_point(0, sum(self.timesteps) / len(self.timesteps), "", "Timesteps", "Average Timesteps Lasted")
        graph_plot_point(0, self.crashed.count(0) / len(self.crashed) * 100, "", "Success Rate %", "Average Success Rate")

        self.writer.flush()
        self.writer.close()

    def make_plot(self, data, title, y_end, y_tick_end, y_tick_amt):
        fig, ax = plt.subplots()
        ax.bar([f"{title}"], data)
                
        ax.set_yticks(range(0, y_tick_end, y_tick_amt))

        # Enable tick marks
        ax.tick_params(axis='both', which='both', length=6)

        ax.set_ylim(0, y_end)
        self.writer.add_figure(f"{title}", fig, 0)
        plt.close(fig)

    def complete_episode(self):
        self.env = gym.make('highway-v0', render_mode="rgb_array", config=self.config)

        self.use_gpt = 0
        state, _ = self.env.reset()
        frame_count = 0
        obs = state
        done = False
        speeds = []

        # Create folder to save frame images to
        frames = f"{self.folder_name}/{self.settings['output_subfolder']}_{int(time.time())}"

        # Create new empty folder to save each frame of the environment
        os.makedirs(frames, exist_ok=True)

        # Create file to log output to
        log = open(f"{frames}/log.txt", "w")

        while not done:
            # Save frame to frame list
            frame = self.env.render()
            if self.settings["save_frame_images"] == "True":
                Image.fromarray(frame).save(f"{frames}/frame_{frame_count:05d}.png")

            cars = [[Vehicle(name="Ego vehicle", x_pos=round(obs[0][1], 4), lane=round(obs[0][2] / 4 + 1), x_vel=round(obs[0][3], 4), y_vel=round(obs[0][4], 4))]]
            speeds.append(round(obs[0][3], 4))

            for i in range (1, len(obs[0])):
                cars.append([Vehicle(name=f"Vehicle: {i}", x_pos=round(obs[i][1], 4), lane=round(obs[i][2] / 4 + 1), x_vel=round(obs[i][3], 4), y_vel=round(obs[i][4], 4))])

            # Initialize log with output of all cars
            log.write(f"Timestep: {frame_count}\n\n")
            frame_count += 1
            for i in range (0, len(cars)):
                log.write(f"{cars[i]}\n")

            log.write("\n")

            if self.use_gpt:
            # Prompt ChatGPT with list of cars and background prompt, then take the specified action
                response, action = ask_chat_gpt(prompt, ACTIONS_ALL, cars, "")
                log.write(f"Action: {return_action(action)}\n\n")
                log.write(f"Response:\n{response}\n\n")
                next_state, _, terminated, truncated, _ = self.env.step(action)
            else:
                next_state, _, terminated, truncated, _ = self.env.step(1)

            obs = next_state
            done = terminated or truncated

            log.write("-------------------------------------------------\n\n")

        # After episode ends save the last frame
        # This is necessary to save the frame of a crash
        frame = self.env.render()
        if self.settings["save_frame_images"] == "True":
            Image.fromarray(frame).save(f"{frames}/frame_{frame_count:05d}.png")

        log.close()
        self.env.close()

        return frame_count, speeds

    def modify_general(self):
        done = 0
        while 1:
            print("Current settings:")
            settings_list = {}
            count = 1
            for thing in self.settings:
                settings_list[count] = (thing, self.settings[thing])
                count += 1

            print("0: Exit")
            for thing in settings_list:
                print(f"{thing}: {settings_list[thing][0]} = {settings_list[thing][1]}")
            print()

#            while 1:
            val = input("Which setting would you like to modify?: ")

            try:
                val = int(val)
                if(val < 0 or val > len(settings_list)):
                    print("Number entered is out of valid range\n")
                else:
                    if val == 0:
                        return
                    print()
                    if settings_list[val][0] == "save_frame_images":
                        self.modify_general_item_set(settings_list[val], {"1": "True", "2": "False"})
                    else:
                        self.modify_general_item(settings_list[val])
#                   done = 1
            except:
                print("Textual input is not valid\n")

            time.sleep(1)

    def modify_general_item_set(self, setting, options):
        print(f"{setting[0]} is currently {setting[1]}")
        while 1:
            print("What would you like to change it to? ", end="")
            for key in options:
                print(f"{key}: {options[key]} ", end="")
            val = input("\nYour response: ")

            print()
            print(f"You entered: '{options[val]}'")
            done = input("Is this correct? Yes: 1 No: 2\nYour response: ")
            print()
            if done == "1":
                break

        self.settings[setting[0]] = options[val]

    def modify_general_item(self, setting):
        print(f"{setting[0]} is currently {setting[1]}")
        while 1:
            val = input("What would you like to change it to?: ")
            print()
            print(f"You entered: '{val}'")
            done = input("Is this correct? Yes: 1 No: 2\nResponse: ")
            print()
            if done == "1":
                break

        self.settings[setting[0]] = val

    def modify_graphs(self):
        pass

sim = Simulator()

while(1):
    val = input("0: Exit\n1: General settings\n2: Graphs\n3: Test agent\nYour choice: ")

    if val == "0":
        break

    print()
    
    match val:
        case "1":
            sim.modify_general()
        case "2":
            sim.modify_graphs()
        case "3":
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