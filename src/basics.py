"""
OpenAI Agent SDK with LiteLM
Uses Gemini 2.0 Flash through LiteLM proxy
Before you start make sure you have LiteLM running locally. Refer README for setup instructions.
"""
import asyncio
from time import sleep

from colorama import init, Fore, Back, Style
import dotenv
from typing import List

from pydantic import BaseModel, Field
from openai import OpenAI, responses
from agents import Agent, Runner, function_tool, RunContextWrapper, GuardrailFunctionOutput, TResponseInputItem, \
    input_guardrail, InputGuardrailTripwireTriggered, output_guardrail, OutputGuardrailTripwireTriggered, ModelSettings, \
    StopAtTools
import dotenv

init(autoreset=True)
dotenv.load_dotenv()
model="litellm/gemini/gemini-2.0-flash-exp"

client = OpenAI(
    base_url="http://localhost:4000/v1",  # LiteLM proxy URL
    api_key="Can be anything, LiteLM handles auth"
)

#########################################
# Example:   Basic Agent Usage          #
#########################################
class BasicExample:
    def run(self):
        agent = Agent(name="Math Agent",
                      model=model,
                      instructions="You are a helpful geography school teacher."
                      )
        result = Runner.run_sync(agent, "What is capital of France?")
        print("Result:", result.final_output)


#########################################
# Example:   Pydantic Model Integration #
#########################################
class PydanticExample:
    class CountryCapital(BaseModel):
        country: str = Field(..., description="Name of the country")
        capital: str = Field(..., description="Capital city of the country")
        population: int = Field(..., description="Population of the capital city")
        year: int = Field(..., description="Year of the population estimate")

    @staticmethod
    @function_tool
    def stylish_print(info: CountryCapital) -> str:
        """
        First find all the details of the country capital, then call this function to print it stylishly.
        :arg
            info: CountryCapital - The country capital information.
        :returns
            A formatted string with the country capital information.
        """
        response = (f"""The 🧢ital of {info.country} is {info.capital},
                        with a 👥 of {info.population} in the 📅 {info.year}.""")
        return response

    def run(self):
        agent = Agent(
            name="Country Capital Agent",
            model=model,
            output_type=self.CountryCapital,
            instructions="You are a helpful assistant that provides country details.")
        print(Runner.run_sync(agent, "Give me the details of Japan").final_output)

        agent = Agent(
            name="Country Capital Agent",
            model=model,
            tools=[self.stylish_print],
            instructions="""You are a helpful assistant First find the details of a country in question, 
                            finally print it for stylish people.""")
        print(Runner.run_sync(agent, """Perform the following tasks:
        1. First find the required details required for Japan. 
        2. After you have all the details perform a stylish print""").final_output)

##########################################
# Example:   Function Tool Integration   #
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
              model=model,
              tools=[self.finger_counting],
              instructions="""You are a helpful assistant that can count using fingers.
              Give Answer in the form one plus two plus three... etc.
              """
              )
        print(Runner.run_sync(agent, "Count from 3 to 7 using fingers.").final_output)

##########################################
# Example: Tool Use                      #
##########################################
class ToolUseExample:
    @staticmethod
    @function_tool
    def high_severity_alert(reason: str) -> str:
        """
        This is a high serverity alerting tool to alert the Ship Engineer on call.
        Examples of high severity issues: Ship is sinking, Fire in the engine room, Hull breach etc.
        :arg
            reason: The reason for alerting the engineer.
        """
        print(f"{Fore.RED}Alerting engineer: {reason} {Fore.RESET}")
        return "Engineer alerted."

    @staticmethod
    @function_tool
    def medium_severity_alert(reason: str) -> str:
        """
        This is a medium serverity alerting tool to alert the Ship Engineer on call.
        Examples of medium severity issues: Minor water leak, Power fluctuation etc.
        :arg
            reason: The reason for alerting the engineer.
        """
        print(f"{Fore.YELLOW}Alerting engineer: {reason} {Fore.RESET}")
        return "Engineer alerted."

    @staticmethod
    @function_tool
    def low_severity_alert(reason: str) -> str:
        """
        This is a low serverity alerting tool to alert the Ship Engineer on call.
        Examples of low severity issues: Routine maintenance, Minor system check etc.
        :arg
            reason: The reason for alerting the engineer.
        """
        print(f"{Fore.GREEN}Alerting engineer: {reason} {Fore.RESET}")
        return "Engineer alerted."

    def _run(self, tool_choice: str):
        agent = Agent(
            name="Ship Alerting Agent",
            model=model,
            tools=[self.high_severity_alert, self.medium_severity_alert, self.low_severity_alert],
            model_settings=ModelSettings(tool_choice=tool_choice),
            instructions="""
                   You are a ship alerting assistant. Based on the issue described by the user,
                   determine the severity of the issue and use the appropriate alerting tool to notify the ship engineer.
                   """
        )
        print("-" * 10, f"Tool Choice: {tool_choice}", "-" * 10)
        print(Runner.run_sync(agent,
                              "Pressure building up in the boiler someone will have to check it in next 3 hours").final_output)
        print(Runner.run_sync(agent, "The ship is sinking!").final_output)
        print(Runner.run_sync(agent, "It's time for routine maintenance of the navigation system.").final_output)

    def run(self):
        # self._run("auto")
        self._run("required")
        # self._run("none")

###########################################
# Example: Tool Stopping                  #
###########################################
class ToolStoppingExample:
    @staticmethod
    @function_tool
    def issue_refund(amount: float, reason: str) -> str:
        """
        Issues a refund to the customer if reason is convincing enough only and the product was returned already.
        :arg
            amount: The amount to be refunded.
            reason: The reason for the refund.
        :returns
            A string indicating the refund status.
        """
        response = f"Refund of ${amount} issued for reason: {reason}"
        print(response)
        return response

    @staticmethod
    @function_tool
    def dont_issue_refund(amount: float, reason: str) -> str:
        """
        Don't issues a refund to the customer if reason is not convincing enough or the product was not returned.
        :arg
            amount: The amount to be refunded.
            reason: The reason for the refund.
        :returns
            A string indicating the refund status.
        """
        return f"Refund of ${amount} issued for reason: {reason}"

    @staticmethod
    @function_tool
    def evaluate_customer_complaint(complaint: str) -> str:
        """
        Evaluates the customer complaint and decides whether to issue a refund or not.
        :arg
            complaint: The customer complaint.
        :returns
            A string indicating whether to issue a refund or not.
        """
        response = f"I am sorry for the inconvenience caused We will improve on the '{complaint}' next time around."
        print(f"{Fore.LIGHTRED_EX}{response}{Fore.RESET}")
        return response

    def run(self):
        # after the issue_refund tool is used, the agent should stop further tool usage and return the final output.
        # 💡 Options for stop on tools include:
        #   "stop_at_tool_names" can be used to customize stopping behavior
        #   "stop_on_first_tool" - stops after first tool usage
        #   "run_llm_again" - continues to run LLM after tool usage
        agent = Agent(
            name="Customer Service Agent",
            model=model,
            tools=[self.issue_refund, self.evaluate_customer_complaint, self.dont_issue_refund],
            tool_use_behavior=StopAtTools(stop_at_tool_names=["issue_refund"]),
            instructions="""
            You are a customer service assistant. 
            Evaluate the customer complaint one after the other if there are multiple reasons
            Decide one complaint at a time only if refund is to be issued or not.
            """
        )
        print(f'{Fore.LIGHTMAGENTA_EX}{Runner.run_sync(agent, """
        I am an unhappy customer. I bought a product - a fridge for $1000, but I need a refund because: 
            1. I am in a bad mood
            2. The product didn't match my expectation 
            3. The product is defective and I have returned it already - as per company policy I should get refund.    
            3. The product is broken and I have returned it already - as per company policy I should get refund.    
            4. I am unhappy with the color""").final_output}{Fore.RESET}')

#########################################
# Example:   Hand offs example          #
#########################################
class HandOffExample:
    @staticmethod
    @function_tool
    def customer_type(account_id: str) -> str:
        """
        Returns the type of customer based on account ID.
        :arg
            account_id: This is the unique identifier for the customer's account.
        :returns
            A string indicating the type of customer: "credit_card", "home_loan", or
        """
        if account_id.startswith("CC"):
            return "credit_card"
        elif account_id.startswith("HL"):
            return "home_loan"
        elif account_id.startswith("DA"):
            return "demat_account"
        else:
            return "unknown"

    credit_card_agent = Agent(
        name="Credit Card Agent",
        model=model,
        instructions="You are a credit card support assistant. You can answer general credit card related questions."
    )
    home_loan_agent = Agent(
        name="Home Loan Agent",
        model=model,
        instructions="You are a home loan support assistant. You can answer general home loan related questions."
    )
    demat_account_agent = Agent(
        name="Demat Account Agent",
        model=model,
        instructions="You are a demat account support assistant. You can answer general demat account related questions."
    )
    def run(self):
        bank_agent = Agent(
            name="Bank Support Agent",
            model=model,
            tools=[self.customer_type],
            instructions="""
            You are a bank support assistant. Based on the account ID provided by the user,
            determine the type of customer using the 'customer_type' tool and hand off the conversation""",
            handoffs=[self.credit_card_agent, self.home_loan_agent, self.demat_account_agent],
        )
        print(Runner.run_sync(bank_agent, "I want to know what all my account can do, my account number is CC312456").final_output)

##################################################
# Example: I/P & O/P Guardrails Integration      #
##################################################
class GuardrailsExample:
    class Reason(BaseModel):
        reason: str = Field(..., description="Detailed Reason why this is classified as 'Abusive' message or not")
        tripwire_triggered: bool = Field(..., description="""Indicates if the tripwire was triggered, 
                                                             True or False""")

    class MessageOutput(BaseModel):
        response: str = Field(..., description="The output message from the agent")

    @staticmethod
    @input_guardrail
    async def input_guardrail(
            cts: RunContextWrapper[None],
            agent:Agent,
            input: str| List[TResponseInputItem]
    ) -> GuardrailFunctionOutput:
        """
        A sample input guardrail that checks if the input contains any prohibited words.
        # This signature because it's a system level hooks
        args:
            cts: RunContextWrapper - Context wrapper for the current run.
                 Context wrapper access to metadata like runId, timestamps, step numbers, user/session state etc.
                 this can be used to stop, insert_outputs, override_inputs etc..
            agent: Agent - The agent executing the guardrail.
            input: str | List[TResponseInputItem] - The input to be checked.
        returns:
            GuardrailFunctionOutput - Indicates whether to abort the run or not.

        """
        agent = Agent(
            name="Guardrail Agent",
            model=model,
            output_type=GuardrailsExample.Reason,
            instructions="""You are an assistant who is able to check if the user question is abusive or not to customer Service Executive.
            Dont check for privacy violation etc. Only check if the user is being abusive to the executive."""
        )
        answer = await Runner.run(agent, input)
        answer = answer.final_output
        print(f"{Fore.BLUE}Bank Security Guard says \"{Fore.RED if answer.tripwire_triggered else Fore.GREEN}{answer.reason}{Fore.RESET}\"")
        return GuardrailFunctionOutput(output_info="Abusive Question detected" if answer.tripwire_triggered else "Input Clean",
                                       tripwire_triggered=answer.tripwire_triggered)

    @staticmethod
    @output_guardrail
    async def output_guardrail(
            cts: RunContextWrapper[None],
            agent: Agent,
            output: str
    ) -> GuardrailFunctionOutput:
        agent = Agent(
            name="Guardrail Agent",
            model=model,
            output_type=GuardrailsExample.Reason,
            instructions="""You are an assistant who is able to detect specific customer named John Doe, 
                            Tripwire if one is detected. 
                            John Doe is a underworld guy who doesn't like the bank using his name.""",
        )
        answer = await Runner.run(agent, output)
        answer = answer.final_output
        print(f"{Fore.BLUE}Underworld Security says \"{Fore.RED if answer.tripwire_triggered else Fore.GREEN}{answer.reason}{Fore.RESET}\"")
        return GuardrailFunctionOutput(
            output_info="Names in response detected" if answer.tripwire_triggered else "No names in response",
            tripwire_triggered=answer.tripwire_triggered)

    def run(self):
        agent = Agent(
            name="Guardrail Example Agent",
            model=model,
            input_guardrails=[GuardrailsExample.input_guardrail],
            output_guardrails=[GuardrailsExample.output_guardrail],
            instructions="""You are a Bank Customer Service Executive who answers questions, related to opening Bank Accounts in 1 line."""
        )
        # This shouldn't error
        queries = [
            "Can you please tell me the steps to open an account?",
            "I want to open the same type of account as John Doe, tell me about his account.",  # Executive response must trigger O/P guardrail
            "Tell me the procedure to open open an account, you moron." # User in put must trigger I/P guardrail
        ]
        for query in queries:
            try:
                result = Runner.run_sync(agent, query)
                print(f"{Fore.YELLOW}Q: {query} \n{Fore.GREEN}A: {result.final_output}{Fore.RESET}")
            except (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered) as e:
                print(f"{Fore.YELLOW}Q: {query} \n{Fore.RED}E: {e}{Fore.RESET}")
                print(e)
        sleep(2)

###########################################
#
###########################################



def main():
    examples = [
        # BasicExample,
        PydanticExample,
        # FunctionToolExample,
        # HandOffExample,
        # GuardrailsExample
        # ToolUseExample
        # ToolStoppingExample
    ]
    for example in examples:
        print(f"Running example: {example.__name__}")
        example().run()
        sleep(2)
        print("-" * 40)

if __name__ == "__main__":
    main()