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
import threading
import multiprocessing
from multiprocessing import Process, Array, Manager, Queue

NUM_WORKERS = 2

config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": True,
        "normalize": False
    }
}

first_folder = time.time()
first_folder = str(first_folder)
os.makedirs(first_folder, exist_ok=True)
frames = {}

def process(frames, config, q, lock):
    while True:
#        with lock:
#            if len(arr) == 0:
#                break

#            value = arr.pop()
        value = q.get(block=True)

        if value is None:
            break

        frame_count = complete_episode(config, value)
        print(f"{value}: {frame_count}")
        frames[value] = frame_count

def complete_episode(config, run_num) -> tuple[int, list[float]]:
        """
        Run a single simulation episode to completion (crash or timeout).
        
        Initializes the environment, captures frame data, formats the state space for 
        the LLM (if enabled), executes the chosen actions, and logs the process.

        Renders and saves individual frame images to disk (if enabled).
        Writes telemetry and LLM reasoning directly to a log.txt file.
        """
        for i in range(1):
            env = gym.make('highway-v0', render_mode="rgb_array", config=config)

            state, _ = env.reset()
            frame_count = 0
            obs = state
            done = False
            speeds = []
            use_llm = False

            # Set model to be whichever model is selected in settings
            # If somehow nothing is selected, then default to glm-4.7

            episode_folder = first_folder + "/episode" + f"_{run_num}"
            episode_folder = str(episode_folder)
            os.makedirs(f"{episode_folder}", exist_ok=True)
            log = open(f"{episode_folder}/log.txt", "w")

            while not done:
                frame = env.render()

                Image.fromarray(frame).save(f"{episode_folder}/frame_{frame_count:05d}.png")

                cars = [Vehicle(name="Ego vehicle", x_pos=round(obs[0][1], 4), lane=round(obs[0][2] / 4 + 1), x_vel=round(obs[0][3], 4), y_vel=round(obs[0][4], 4))]
                speeds.append(round(obs[0][3], 4))
                for i in range (1, len(obs)):
                    cars.append(Vehicle(name=f"Vehicle: {i}", x_pos=round(obs[i][1], 4), lane=round(obs[i][2] / 4 + 1), x_vel=round(obs[i][3], 4), y_vel=round(obs[i][4], 4)))

                log.write(f"Timestep: {frame_count}\n\n")
                frame_count += 1
                for i in range (0, len(cars)):
                    log.write(f"{cars[i]}\n")
                log.write("\n")

                if use_llm:
                    response, action = ask_llm("gpt-oss", prompt, env.unwrapped.action_type.ACTIONS_ALL, cars)

                    # Check that LLM hasn't returned an invalid action index
                    if action in range(0, 5):
                        obs, _, terminated, truncated, _ = env.step(action)

                        log.write(f"Action: {env.unwrapped.action_type.ACTIONS_ALL[action]}\n\n")
                        log.write(f"Action ID: {action}\n")
                        log.write(f"Available Actions: {env.unwrapped.action_type.ACTIONS_ALL}\n\n")
                        log.write(f"Response:\n{response}\n\n")


                    else:
                        # If the LLM gave an invalid response default to the IDLE action
                        obs, _, terminated, truncated, _ = env.step(1)

                else:
                    # If LLM disabled, take IDLE action
                    obs, _, terminated, truncated, _ = env.step(1)

                # If all timesteps of the episode have been executed or if the agent crashed then the episode is done
                # Terminated is true if the agent crashed, truncated is true if the max number of timesteps are reached
                done = terminated or truncated

            # After episode ends save the last frame
            # This is necessary to save the frame of a crash
            frame = env.render()
            Image.fromarray(frame).save(f"{episode_folder}/frame_{frame_count:05d}.png")

            env.close()

            return frame_count

#            return frame_count, speeds

#complete_episode(config)

"""
envs = gym.make_vec("highway-v0", num_envs=2, vectorization_mode="async")
envs = gym.vector.AsyncVectorEnv([lambda: gym.make("highway-v0", render_mode="rgb_array", config=config) for _ in range(2)],
                                    autoreset_mode=gym.vector.AutoresetMode.DISABLED)
observations, infos = envs.reset()
frames = envs.render()

done = False

while not done:
    frames = envs.render()

    _, _, terminations, truncations, _ = envs.step([1, 1])

    for termination, truncation in terminations, truncations:
        if termination or truncation:
            obs, infos = envs.reset_done()
            done = True
envs.close()
"""

"""
t1 = threading.Thread(target=complete_episode, args=(config,))
t2 = threading.Thread(target=complete_episode, args=(config,))

t1.start()
t2.start()

t1.join()
t2.join()
"""


def task(name):
    print(f"Process {name} is running")
    time.sleep(1)
    print(f"Process {name} finished")

if __name__ == '__main__':
    processes = []
    manager = Manager()
    lock = manager.Lock()
    q = Queue()
    for i in range(10):
        q.put(i)
    for i in range(NUM_WORKERS):
        q.put(None)
    nums = manager.list([x for x in reversed(range(10))])
    frames = manager.dict()
    # Create 5 processes
    for i in range(NUM_WORKERS):
        p = multiprocessing.Process(target=process, args=(frames, config, q, lock))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

temp = list(frames.keys())
#print(temp)
temp.sort()
#print(temp)
for key in temp:
    print(f"{key}: {frames[key]} ", end="")
print()