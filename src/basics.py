"""
OpenAI Agent SDK with LiteLM
Uses Gemini 2.0 Flash through LiteLM proxy
"""
from pydantic import BaseModel, Field
from openai import OpenAI
from agents import Agent, Runner, function_tool
import dotenv
dotenv.load_dotenv()

client = OpenAI(
    base_url="http://localhost:4000/v1",  # LiteLM proxy URL
    api_key="Can be anything, LiteLM handles auth"
)

#########################################
# Example 1: Basic Agent Usage          #
#########################################
class BasicExample:
    def run(self):
        agent = Agent(name="Math Agent",
                      model="litellm/gemini/gemini-2.0-flash-exp",
                      instructions="You are a helpful math assistant."
                      )
        result = Runner.run_sync(agent, "What is capital of France?")
        print("Result:", result.final_output)

#########################################
# Example 2: Pydantic Model Integration #
#########################################
class PydanticExample:
    class CountryCapital(BaseModel):
        country: str = Field(..., description="Name of the country")
        capital: str = Field(..., description="Capital city of the country")
        population: int = Field(..., description="Population of the capital city")
        year: int = Field(..., description="Year of the population estimate")

    def run(self):
        agent = Agent(name="Country Capital Agent",
                      model="litellm/gemini/gemini-2.0-flash-exp",
                      output_type=self.CountryCapital,
                      instructions="You are a helpful assistant that provides country capitals.")
        print(Runner.run_sync(agent, "What is the capital of Japan and its population in 2025?").final_output)

##########################################
# Example 3: Function Tool Integration   #
##########################################
class FunctionToolExample:
    @staticmethod
    @function_tool
    def finger_counting(a: int, b: int) -> int:
        """
        Counts from a to a+b using fingers.
        """
        print("Counting on finger...")
        till_now = a
        for i in range(b):
            till_now += 1
            print(f"{till_now}...")
        return till_now

    def run(self):
        agent = Agent(name="Function Tool Agent",
              model="litellm/gemini/gemini-2.0-flash-exp",
              tools=[self.finger_counting],
              instructions="""You are a helpful assistant that can count using fingers.
              Give Answer in the form one plus two plus three... etc.
              """
              )
        print(Runner.run_sync(agent, "Count from 3 to 7 using fingers.").final_output)

def main():
    examples = [BasicExample, PydanticExample, FunctionToolExample]
    for example in examples:
        print(f"Running example: {example.__name__}")
        example().run()
        print("-" * 40)

if __name__ == "__main__":
    main()