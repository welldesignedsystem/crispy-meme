"""
OpenAI Agent SDK with LiteLM
Uses Gemini 2.0 Flash through LiteLM proxy
"""

from openai import OpenAI
from agents import Agent, Runner
import dotenv
dotenv.load_dotenv()

client = OpenAI(
    base_url="http://localhost:4000/v1",  # LiteLM proxy URL
    api_key="Can be anything, LiteLM handles auth"
)
def main():
    agent = Agent(name="Math Agent",
                  model="litellm/gemini/gemini-2.0-flash-exp",
                  instructions="You are a helpful math assistant.")
    result = Runner.run_sync(agent, "What is capital of France?")
    print("Result:", result.final_output)

if __name__ == "__main__":
    main()