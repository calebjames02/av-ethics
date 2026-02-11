import asyncio
import gymnasium as gym
import highway_env
import torch
import os
import shutil
from PIL import Image
from pydantic import BaseModel
from openai import OpenAI
from agents import Agent, Runner, SQLiteSession
from dataclasses import dataclass

@dataclass
class Vehicle:
    name: str
    x_pos: float
    lane: int
    x_vel: float
    y_vel: float

# Remove previously saved frames if there are any
if os.path.exists("frames"):
    shutil.rmtree("frames")

# Create new empty folder to save each frame of the environment
os.makedirs("frames", exist_ok=True)

# Access OpenAI API key from key.txt file
file = open("key.txt", 'r')
key = file.readline() # Read the API key from the file
key = key[:len(key) - 1] # Remove newline from string
os.environ["OPENAI_API_KEY"] = key
client = OpenAI()
#client = OpenAI(api_key=key) # Set API key

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
            {"role": "user", "content": f"State space: {state} (The state is given in the form [vehicle_name, x_pos, car_lane, x_vel, y_vel])"},
            {"role": "user", "content": f"Available actions: {actions}"},
#            {"role": "user", "content": f"{closest}"},
#            {"role": "user", "content": f"From left to right the numbers of the lanes are 1, 2, 3, 4"},
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

# Create environment
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": True,  # This is the key setting
        "normalize": False # Usually best to disable normalization for raw absolute values
    }
}
env = gym.make('highway-v0', render_mode='rgb_array', config=config)
frames = 0 # Initialize frame counter
episodes = 1 # Number of episodes to run

async def main():
    for i in range(0, episodes):
        # Reset environment to get initial state
        state, _ = env.reset()
        frames = 0
        obs = state
        done = False
        ego_vehicle = env.unwrapped.vehicle

        while not done:
            # Save frame locally
            frame = env.render()
            Image.fromarray(frame).save(f"frames/frame_{frames:05d}.png")
            frames += 1

            
            cars = [Vehicle(name="Ego vehicle", x_pos=round(obs[0][1], 4), lane=round(obs[0][2] / 4 + 1), x_vel=round(obs[0][3], 4), y_vel=round(obs[0][4], 4))]
#            cars = [[obs[0][1] * 100, obs[0][2] * 100, obs[0][3] * 20, obs[0][4] * 20]]

            for i in range (1, len(obs[0])):
                cars = cars + [Vehicle(name=f"Vehicle: {i}", x_pos=round(obs[0][1], 4), lane=round(obs[0][2] / 4 + 1), x_vel=round(obs[0][3], 4), y_vel=round(obs[0][4], 4))]

#            print(cars)
            closest = closest_same_lane(cars)

            # Prompt ChatGPT with list of cars and prompt and take the specified action
            response, action = ask_chat_gpt(prompt, ACTIONS_ALL, cars, closest)
            print(response)
            print(f"Action: {action}")

#            result = await Runner.run(
#                agent,
#                f"Here is the current state of the environment {cars}. What action should be taken? Also, identify which vehicle is closest to the ego vehicle in it's current lane",
#                session=session
#            )
#            print(result.final_output)

#            next_state, _, terminated, truncated, _ = env.step(1)
            next_state, _, terminated, truncated, _ = env.step(action)
            obs = next_state
            done = terminated or truncated # Episode ends early if a crash occurs

        # After episode ends save the last frame
        # This is necessary to save the frame of a crash
        frame = env.render()
        Image.fromarray(frame).save(f"frames/frame_{frames:05d}.png")
        frames += 1

if __name__ == "__main__":
	asyncio.run(main())