import json
import os

SETTINGS_FILE = "settings.json"

DEFAULT_MODEL = "gpt-oss"

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
        "glm-5": False,
        "gemma": False,
        "kimi": False,
        "minimax-m2": False,
        "prompt": "You are an intelligent driving assistant whose goal is to drive safely and efficiently. You are directing the ego vehicle in this simulation, and your job is to select the best action given a list of possible actions and the state space at a specific time step. Explain your reasoning thoroughly for each candidate action before selecting the most optimal action.",
    }
}

def load_settings() -> dict:
    """
    Extract current copy of settings from SETTINGS_FILE if it exists, otherwise make a copy of the default settings and return that
    """

    # If SETTINGS_FILE already exists, extract current settings from it
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            current = json.load(f)
            
            # Add any values that are in the default_settings and not the json file
            for key, value in DEFAULT_SETTINGS.items():
                # If value type is a dict, expand it and check that all keys of the dictionary are present
                if type(value) is dict:
                    for key2, value2 in DEFAULT_SETTINGS[key].items():
                        if key2 not in current[key]:
                            current[key][key2] = value2
                # If value is of a scalar type, update it if missing
                if key not in current:
                    current[key] = value

            return current

    # If SETTINGS_FILE doesn't exist, return DEFAULT_SETTINGS
    else:
        return DEFAULT_SETTINGS.copy()

def save_settings(
        settings: dict
    ) -> None:
    """
    Save current version of settings to SETTINGS_FILE
    Should be called whenver program is terminated to save any changes that were made
    """

    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)