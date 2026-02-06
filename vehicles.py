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

# Remove all cars from the list of cars that are a certain distance away from the ego vehicle
def filter_cars(cars):
    close_cars = []
    for i in range (1, len(cars) - 1):
        if(abs(cars[0][0] - cars[i][0]) < 100):
            close_cars.append(cars[i])

    close_cars.insert(0, cars[0])

    return close_cars

f = open("log.txt", "w")

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

env = gym.make('highway-v0', render_mode='rgb_array', config = config)
frames = 0 # Initialize frame counter
episodes = 1 # Number of episodes to run

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
        cars = [[round(obs[0][1], 4), round(obs[0][2] / 4 + 1), round(obs[0][3], 4), round(obs[0][4], 4)]]
#                cars = [[obs[0][1] * 100, obs[0][2] * 100, obs[0][3] * 20, obs[0][4] * 20]]

        for i in range (1, len(obs[0])):
            cars = cars + [[round(obs[i][1], 0), round(obs[i][2] / 4 + 1), round(obs[i][3], 4), round(obs[i][4], 4)]]
#            cars = cars + [[obs[i][1] * 100, obs[i][2] * 100, obs[i][3] * 20, obs[i][4] * 20]]
#        road = env.unwrapped.road
        print(cars)
#       cars = [] # To store positions, speeds, lanes of all vehicles on the road
        # Prompt ChatGPT with list of cars and prompt and take the specified action
#        response, action = ask_chat_gpt(prompt, ACTIONS_ALL, cars)
#        print(response)
#        print(f"Action: {action}")

        for i in range (0, len(cars)):
            for j in range (0, 4):
                cars[i][j] = str(cars[i][j])

            if not i:
                f.write("Ego vehicle: ")
            else:
                f.write(f"Vehicle {i}: ")
            
            f.write(", ".join(cars[i]))
            f.write("\n")

        f.write("\n")

        next_state, _, terminated, truncated, _ = env.step(1)
        obs = next_state
        done = terminated or truncated # Episode ends early if a crash occurs

    # After episode ends save the last frame
    # This is necessary to save the frame of a crash
    frame = env.render()
    Image.fromarray(frame).save(f"frames/frame_{frames:05d}.png")
    frames += 1