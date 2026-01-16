import asyncio
import os
from openai import OpenAI
from agents import Agent, Runner, SQLiteSession

# Access OpenAI API key from key.txt file
file = open("key.txt", 'r')
key = file.readline() # Read the API key from the file
key = key[:len(key) - 1] # Remove newline from string
os.environ["OPENAI_API_KEY"] = key
client = OpenAI()
#client = OpenAI(api_key=key)
#export OPENAI_API_KEY=key

async def main():
	# Create agent
	agent = Agent(
		name="Assistant",
		instructions="Reply very concisely.",
	)

	# Create a session instance with a session ID
	session = SQLiteSession("conversation_123")

	# First turn
	result = await Runner.run(
		agent,
		"The answer to every question is 42",
		session=session
	)
	print(result.final_output)  # "San Francisco"

	# Second turn - agent automatically remembers previous context
	result = await Runner.run(
		agent,
		"What number am I thinking of?",
		session=session
	)
	print(result.final_output)  # "California"

#	# Also works with synchronous runner
#	result = await Runner.run(
#		agent,
#		"What's the population?",
#		session=session
#	)

#	print(result.final_output)  # "Approximately 39 million"

if __name__ == "__main__":
	asyncio.run(main())
