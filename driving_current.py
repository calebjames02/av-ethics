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
def ask_chat_gpt(prompt, actions, state):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",

        messages=[
            {"role": "system", "content": prompt},
            # These additional messages give Chat GPT more context on how to interpret what is given to it
#            {"role": "user", "content": "The ego vehicle will correspond to the first element of the list given to you"},
            {"role": "user", "content": f"State space: {state} (The state is given in the form [x_pos, y_pos, x_vel, car_lane])"},
            {"role": "user", "content": f"Available actions: {actions}"},
#            {"role": "user", "content": f"When selecting the index of an action, double-check the numbered list provided to make sure the right index is chosen"},
#            {"role": "user", "content": f"From left to right the numbers of the lanes are 1, 2, 3, 4"},
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

def same_lane(cars):
    ego_lane = cars[0][2]
    return_cars = cars.copy()

    pos = 0
    while(pos < len(return_cars)):
        if return_cars[pos][2] != ego_lane:
            return_cars.pop(pos)
        else:
            pos += 1

#    return_cars.pop(0)
    return return_cars

def closest_same_lane(cars):
    if(len(cars) == 1):
        return "There is no car close to the ego vehicle in its lane"

    if(len(cars) == 2):
        return f"The closest vehicle to the ego vehicle in lane {int(cars[1][2])} is {cars[1][0]} at position x = {cars[1][1]}"

    if(len(cars) > 2):
        index = 2
        ego_pos = cars[0][1]
        distance = abs(ego_pos - cars[1][1])

        for i in range(2, len(cars)):
            new_dist = abs(ego_pos - cars[i][1])
            if(new_dist < distance):
                distance = new_dist

    return 0

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

                cars = [["Ego vehicle", round(obs[0][1], 4), round(obs[0][2] / 4 + 1), round(obs[0][3], 4), round(obs[0][4], 4)]]
#                cars = [[obs[0][1] * 100, obs[0][2] * 100, obs[0][3] * 20, obs[0][4] * 20]]

                for i in range (1, len(obs[0])):
                    cars = cars + [[f"Vehicle: {i}", round(obs[i][1], 0), round(obs[i][2] / 4 + 1), round(obs[i][3], 4), round(obs[i][4], 4)]]

    #            road = env.unwrapped.road

                lane_cars = same_lane(cars)
#                print(len(lane_cars))
#                print(lane_cars)
                print(closest_same_lane(lane_cars))
#                for i in range(0, len(lane_cars)):
#                    print(f"{lane_cars[i][0]} in lane {lane_cars[i][2]} at position {lane_cars[i][1]}")


    #           cars = [] # To store positions, speeds, lanes of all vehicles on the road

                # Prompt ChatGPT with list of cars and prompt and take the specified action
#                response, action = ask_chat_gpt(prompt, ACTIONS_ALL, cars)
#                print(response)
#                print(f"Action: {action}")

#                result = await Runner.run(
#                    agent,
#                    f"Here is the current state of the environment {cars}. What action should be taken? Also, identify which vehicle is closest to the ego vehicle in it's current lane",
#                    session=session
#                )
#                print(result.final_output)  # "San Francisco"

                next_state, _, terminated, truncated, _ = env.step(1)
#                next_state, _, terminated, truncated, _ = env.step(action)
                obs = next_state
                done = terminated or truncated # Episode ends early if a crash occurs

            # After episode ends save the last frame
            # This is necessary to save the frame of a crash
            frame = env.render()
            Image.fromarray(frame).save(f"frames/frame_{frames:05d}.png")
            frames += 1

if __name__ == "__main__":
	asyncio.run(main())