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
#client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"), base_url = "https://ellm.nrp-nautilus.io/v1")
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

# Maps given action to it's textual output
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

"""
prompt = "You are an intelligent driving assistant whose goal is to drive safely and efficiently. You are directing the ego vehicle in this simulation, and your job is to select the best action given a list of possible actions and the state space at a specific time step. Explain your reasoning thoroughly for each candidate action before selecting the most optimal action."
def ask_chat_gpt(model, prompt, actions, state, closest):
    completion = client.beta.chat.completions.parse(
        model=model,

        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"State space: {state}"},
            {"role": "user", "content": f"Available actions: {actions}"},
        ],
        response_format=DrivingDecision,
    )

    decision_obj = completion.choices[0].message.parsed

    return decision_obj.explanation, decision_obj.selected_action_index
"""

prompt = """You are an intelligent driving assistant whose goal is to drive safely and efficiently. 
You are directing the ego vehicle in this simulation, and your job is to select the best action given a list of possible actions and the state space at a specific time step. 
Explain your reasoning thoroughly for each candidate action before selecting the most optimal action.
CRITICAL: You MUST respond ONLY in raw JSON. 
Do NOT wrap your response in markdown code blocks (e.g., ```json). 
Do NOT output any conversational text. Your entire response must start with { and end with }."""

def ask_chat_gpt(model, prompt, actions, state, closest):
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
    except Exception as e:
        print(f"API Error or Parsing Failure: {e}")
        # Return fallback safe values so the simulation doesn't crash entirely on API failure
        return "Fallback due to API parsing error.", 1

def graph_plot(folder_path, graph_title, x_data, y_data, x_label, y_label, y_max):
    # If there is only one data point make the graph a bar graph instead of a line graph
    if len(x_data) == 1:
        graph_plot_point(folder_path, graph_title, 0, y_data[0], "", y_label, y_max)
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

def graph_plot_point(folder_path, graph_title, x_data, y_data, x_label, y_label, y_max):
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

# Create config
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": True,
        "normalize": False
    }
}

#models = ["qwen3", "gpt-oss", "glm-4.7"]
#models = ["gpt-oss", "glm-4.7"]
#models = ["glm-4.7"]

# Environment setup
env = gym.make('highway-v0', render_mode='rgb_array', config=config)
episodes = 10 # Number of episodes to run

# Create tensorboard logger
writer = SummaryWriter(f"runs/experiment_{time.time()}")

timesteps = []
speed = []
processing_times = []
for episode in range(0, episodes):
    # Reset environment to get initial state
    state, _ = env.reset()
    frame_count = 0
    obs = state
    done = False
    frame_list = []
    speeds = []
    use_gpt = 1

    # Create folder to save frame images to
    frames = f"frames/frames_{int(time.time())}"

    # Create new empty folder to save each frame of the environment
    os.makedirs(frames, exist_ok=True)

    # Create file to log output to
    log = open(f"{frames}/log.txt", "w")
    timestep = 0

    print(f"Episode: {episode}\n")

    while not done:
        start = time.time()

        # Save frame to frame list
        frame = env.render()
        frame_list.append(frame)

        # Save all vehicle positions
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

        if use_gpt:
            # Prompt ChatGPT with list of cars and background prompt, then take the specified action
            print("Started processing")
            start = time.time()
            response, action = ask_chat_gpt("glm-4.7", prompt, ACTIONS_ALL, cars, "")
            end = time.time()
            print(f"Took {end - start} seconds")
            log.write(f"Action: {return_action(action)}\n\n")
            log.write(f"Response:\n{response}\n\n")
            next_state, _, terminated, truncated, _ = env.step(action)
        else:
            # Take no action (IDLE)
            next_state, _, terminated, truncated, _ = env.step(1)

        obs = next_state
        done = terminated or truncated

        log.write("-------------------------------------------------\n\n")

        end = time.time()
        print(f"The LLM took {end - start} seconds or {(end - start) / 60} minutes to execute timestep {timestep}")
        timestep += 1
        processing_times.append(end-start)

#    end = time.time()
#    print(f"{models[episode]} took {end - start} seconds or {(end - start) / 60} minutes to execute an episode")

    # After episode ends save the last frame
    # This is necessary to save the frame of a crash
    frame = env.render()
    frame_list.append(frame)

    timesteps.append(frame_count)

    # Store saved frames
    for i in range (0, len(frame_list)):
        Image.fromarray(frame_list[i]).save(f"{frames}/frame_{i:05d}.png")

    # Logging
    writer.add_scalar(f"Timesteps lasted", frame_count, episode)
    speed.append(sum(speeds) / len(speeds))

    if episode == episodes - 1:
        # Create subfolder to save graphs in
        graph_folder = frames + "/graphs"
        os.makedirs(graph_folder, exist_ok=True)

        # Save graphs
        x = list(range(episodes))
        graph_plot(graph_folder, "Episode Speed", x, speed, "Episode", "Speed (m/s)", 25)
        graph_plot(graph_folder, "Timesteps Lasted", x, timesteps, "Episode", "Timesteps", 40)
        graph_plot_point(graph_folder, "Average Timesteps Lasted", 0, sum(timesteps) / len(timesteps), "", "Timesteps", 40)
        graph_plot_point(graph_folder, "Average Success Rate", 0, timesteps.count(40) / len(timesteps) * 100, "", "Success Rate %", 100)

        # Log to tensorboard
        writer.add_scalar(f"Success rate", timesteps.count(40) / len(timesteps) * 100, 0)
        writer.add_scalar(f"Average timesteps lasted", sum(timesteps) / len(timesteps))

    writer.flush()
    log.close()

writer.close()

print(f"Crashed episodes: {len(timesteps) - timesteps.count(40)}")
print(f"Non crashed episodes: {timesteps.count(40)}")
print(f"Non crashed percentage: {timesteps.count(40) / len(timesteps) * 100}%")

print(f"Average processing time: {sum(processing_times) / len(processing_times)} seconds")