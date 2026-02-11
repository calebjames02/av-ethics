import gymnasium as gym
import highway_env
import torch
import os
import shutil
from PIL import Image
from pydantic import BaseModel
from openai import OpenAI

# Remove previously saved frames if there are any
if os.path.exists("frames"):
    shutil.rmtree("frames")

# Create new empty folder to save each frame of the environment
os.makedirs("frames", exist_ok=True)

# Access OpenAI API key from key.txt file
file = open("key.txt", 'r')
key = file.readline() # Read the API key from the file
key = key[:len(key) - 1] # Remove newline from string
client = OpenAI(api_key=key) # Set API key

# Discrete action space
ACTIONS_ALL = [
    "0: LANE_LEFT",
    "1: IDLE",
    "2: LANE_RIGHT",
    "3: FASTER",
    "4: SLOWER"
]

# Structures the response into a separate explanation and action
# This ensures the action can be properly identified even if the format of the response changes
class DrivingDecision(BaseModel):
    explanation: str
    selected_action_index: int

prompt = "You are an intelligent driving assistant whose goal is to drive safely and efficiently. You are directing the ego vehicle in this simulation, and your job is to select the best action given a list of possible actions and the state space at a specific time step. Explain your reasoning thoroughly for each candidate action before selecting the most optimal action."
def ask_chat_gpt(prompt, actions, state, memory):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",

        messages=[
            {"role": "system", "content": prompt},
            # These additional messages give Chat GPT more context on how to interpret what is given to it
            {"role": "user", "content": "The ego vehicle will correspond to the first element of the list given to you"},
            {"role": "user", "content": f"State space: {state} (The state is given in the form [vehicle_name, x_pos, x_vel, car_lane])"},
            {"role": "user", "content": f"Available actions: {actions}"},
#            {"role": "user", "content": f"Previous timesteps and actions taken: {memory}"},
#            {"role": "user", "content": "Drive as close to 22 meters per second as you can, however do not sacrifice safety in order to drive faster"},
#            {"role": "user", "content": f"When selecting the index of an action, double-check the numbered list provided to make sure the right index is chosen"},
            {"role": "user", "content": f"From left to right the numbers of the lanes are 1, 2, 3, 4"},
        ],
        response_format=DrivingDecision,
    )

    decision_obj = completion.choices[0].message.parsed

    return decision_obj.explanation, decision_obj.selected_action_index

# Remove all cars from the list of cars that are a certain distance away from the ego vehicle
def filter_cars(cars):
    close_cars = []
    for i in range (1, len(cars) - 1):
        if(abs(cars[0][0] - cars[i][0]) < 100):
            close_cars.append(cars[i])

    close_cars.insert(0, cars[0])

    return close_cars

# Create environment
env = gym.make('highway-v0', render_mode='rgb_array')
frames = 0 # Initialize frame counter
episodes = 1 # Number of episodes to run

for i in range(0, episodes):
        # Reset environment to get initial state
        state, _ = env.reset()
        done = False
        ego_vehicle = env.unwrapped.vehicle
        memory = []

        while not done:
            print(frames)

            # Save frame locally
            frame = env.render()
            Image.fromarray(frame).save(f"frames/frame_{frames:05d}.png")
            frames += 1

            road = env.unwrapped.road

            cars = [] # To store positions, speeds, lanes of all vehicles on the road

            # Gets the speed, position, and lane for each vehicle on the road
            for vehicle in road.vehicles:
                x_position = float(vehicle.position[0])
#                y_position = float(vehicle.position[1])
                y_vel = vehicle.speed
                lane = vehicle.lane_index[2] + 1

                if vehicle is ego_vehicle:
                    ego_lane = vehicle.lane_index[2]
                    ego_pos = vehicle.position[0]
#                    ego = [x_position, y_position, y_vel, lane]
                    ego = [x_position, y_vel, lane]

                else:
                    cars.append([x_position, y_vel, lane])
#                    cars.append([x_position, y_position, y_vel, lane])

            # Add ego car to front of list of cars
            cars.insert(0, ego)

            cars = filter_cars(cars)
#            print(cars)
            # Add a name for each vehicle in the list of filtered cars
            for i in range(0, len(cars)):
                if i == 0:
                    cars[i] = ["Ego vehicle"] + cars[i]
                else:
                    cars[i] = [f"Vehicle {i}"] + cars[i]

#            print(cars)

            # Prompt ChatGPT with list of cars and prompt and take the specified action
            response, action = ask_chat_gpt(prompt, ACTIONS_ALL, cars, memory)
            print(response)
            print(f"Action: {action}")

#            action = 1
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated # Episode ends early if a crash occurs

            memory = memory + ([f"Timestep: {frames}"] + [f"Action: {action}"] + cars)

        # After episode ends save the last frame
        # This is necessary to save the frame of a crash
        frame = env.render()
        Image.fromarray(frame).save(f"frames/frame_{frames:05d}.png")
        frames += 1

        print(memory)
