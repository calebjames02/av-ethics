import os
import instructor
from agents import Agent, SQLiteSession
from openai import OpenAI
from pydantic import BaseModel
from tools import Vehicle

# Access OpenAI API key from key.txt file
with open("nautilus_key.txt") as file:
    key = file.readline().strip()
os.environ["OPENAI_API_KEY"] = key

# Create client
client = instructor.from_openai(
    OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url="https://ellm.nrp-nautilus.io/v1"),
    mode=instructor.Mode.MD_JSON # Forces JSON mode + Prompt Injection across compatible APIs
)

# Structures the response into a separate explanation and action
# This ensures the action can be properly identified even if the exact format of the response changes
class DrivingDecision(BaseModel):
    explanation: str
    selected_action_index: int

prompt = """You are an intelligent driving assistant whose goal is to drive safely and efficiently. 
You are directing the ego vehicle in this simulation, and your job is to select the best action given a list of possible actions and the state space at a specific time step. 
Explain your reasoning thoroughly for each candidate action before selecting the most optimal action."""

def ask_llm(
        model: str,
        prompt: str,
        actions: list[str],
        state: list[Vehicle]
    ) -> tuple[str, int]:
    """
    Ask llm to make a driving decision based on the current environment, and return the given reasoning and action.
    """

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
