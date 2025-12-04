"""
OpenAI Agent SDK with LiteLM
Uses Gemini 2.0 Flash through LiteLM proxy
Before you start make sure you have LiteLM running locally. Refer README for setup instructions.
"""
import asyncio
from time import sleep

from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from colorama import init, Fore, Back, Style
import dotenv
from typing import List

from pydantic import BaseModel, Field
from openai import OpenAI, responses
from agents import Agent, Runner, function_tool, RunContextWrapper, GuardrailFunctionOutput, TResponseInputItem, \
    input_guardrail, InputGuardrailTripwireTriggered, output_guardrail, OutputGuardrailTripwireTriggered, ModelSettings, \
    StopAtTools, SQLiteSession, handoff
import dotenv
from requests import session

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
    """
    Output:
        (crispy-meme)  ai@gravionis  ~/Code/learn/crispy-meme/src   main ±  uv run basics.py
        Running example: PydanticExample
        country='Japan' capital='Tokyo' population=13960000 year=2023
        Okay, I have performed a stylish print for Japan:
        🧢ital of Japan is Tokyo,with a 👥 of 37435000 in the 📅 2024.
    """
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
        response = f"🧢ital of {info.country} is {info.capital},with a 👥 of {info.population} in the 📅 {info.year}."
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

################################################
#  Example: Function Chaining & Agent As tool  #
################################################
class FunctionChainingExample:
    class CustomerDetails(BaseModel):
        id: int = Field(..., description="Unique identifier for the customer")
        name: str = Field(..., description="Name of the customer")
        age: int = Field(..., description="Age of the customer")
        membership_level: str = Field(..., description="Membership level of the customer")

    @staticmethod
    @function_tool
    def get_customer_details(customer_name: str) -> CustomerDetails:
        """
        Fetches customer details based on the customer name.
        :param customer_name:
        :return:
        """
        membership_level = "Gold" if customer_name.lower() == "daniel" else "Standard"
        return FunctionChainingExample.CustomerDetails(
            id=123,
            name=customer_name,
            age=50,
            membership_level=membership_level
        )

    @staticmethod
    @function_tool
    def get_details_of_promotion_offer(details: CustomerDetails) -> str:
        """
        Sends a promotion email to the customer based on their membership level.
        :param details:
        :return:
        """
        if details.membership_level == "Gold":
            return f"Send Customer {details} the latest offer email."
        else:
            return f"Customer {details} is on standard membership don't send offer email."

    def run(self):
        email_agent = Agent(
            name="Email writing Agent",
            model=model,
            instructions="""You are Funny Sunny an email writing assistant of Funny Marketing Inc. 
            Write a membership benefits email for the customer only if customer is eligible for one.
            Give him or her 10% discount on next purchase as a membership benefit.
            Give a very warm and friendly tone to the email."""
        )

        # Agent as Tool
        email_agent_tool = email_agent.as_tool(
                       tool_name="construct_promotion_email",
                       tool_description="Constructs an email full of Emojis (atleast 50%) for promotion offer for eligible customers.")
        agent = Agent(
            name="Research Agent",
            model=model,
            tools=[self.get_customer_details, self.get_details_of_promotion_offer, email_agent_tool],
            tool_use_behavior=StopAtTools(stop_at_tool_names=["construct_promotion_email"]),
            instructions="""
                You are a customer service assistant. 
                First get the customer details
                get the Customer eligibility for membership benefits
                finally construct the membership benefits email only if customer is eligible.
            """
        )
        print(f'{Runner.run_sync(agent, "Create any membership benefits email body for customer - Daniel.").final_output}')
        print(f'{Runner.run_sync(agent, "Create any membership benefits email body for customer - Thomas.").final_output}')

##########################################
# Example: Short term Memory             #
##########################################
class ShortTermMemoryExample:
    def run(self):
        messages = []
        agent = Agent(
            name="Memory Agent",
            model=model,
            instructions="You are a helpful assistant with memory."
        )
        messages.append({"role": "user", "content": "Hello what is the capital of Antarctica?"})
        messages.append({"role":"assistant", "content": Runner.run_sync(agent, messages).final_output})
        messages.append({"role": "user", "content": "What is its area of land?"})
        messages.append({"role": "assistant", "content": Runner.run_sync(agent, messages).final_output})
        messages.append({"role": "user", "content": "Now tell me its population."})
        messages.append({"role": "assistant", "content": Runner.run_sync(agent, messages).final_output})
        color = Fore.CYAN
        for message in messages:
            match message['role']:
                case 'user': color = Fore.YELLOW
                case 'assistant': color = Fore.GREEN
            print(f"{color}{message['role'].upper()}: {message['content']}{Fore.RESET}")

########################################
# Example: Long Term/Persistent Memory #
########################################
class PersistentMemoryExample:
    def __init__(self):
        self.session = SQLiteSession(
            db_path="../persistent_memory.db",
            session_id="user_1234"
        )
    def run(self):
        agent = Agent(
            name="Persistent Memory Agent",
            model=model,
            instructions="You are a helpful assistant with persistent memory."
        )
        print(Runner.run_sync(agent, "Hello what is the capital of Antarctica?", session=self.session).final_output)
        print(Runner.run_sync(agent, "What is its area of land?", session=self.session).final_output)
        print(Runner.run_sync(agent, "Now tell me its population.", session=self.session).final_output)

########################################
# Example: Deterministic Orchestration #
########################################
class DeterministicOrchestrationExample:
    def run(self):
        bjp_agent = Agent(
            name="BJP Supporter",
            model=model,
            instructions="""
            You are a BJP supporter and you have to answer all questions in favor of BJP.
            Always start with 'As a BJP supporter,' in your answers.
            """,
            model_settings=ModelSettings(temperature=0.0, top_p=1.0)
        )
        congress_agent = Agent(
            name="Congress Supporter",
            model=model,
            instructions="""
            You are a Congress supporter and you have to answer all questions in favor of Congress/INC.
            Always start with 'As a Congress supporter,' in your answers.
            """,
            model_settings=ModelSettings(temperature=0.0, top_p=1.0)
        )
        current_agent = congress_agent
        session = SQLiteSession(
            db_path="../deterministic_memory.db",
            session_id="political_debate")
        for _ in range(6):
            answer = Runner.run_sync(
                current_agent,
                """You have to justify why you are better for India and how the other party is bad. 
                         only provide one sentence at a tim. Always counter the previous point of the other party."""
                ,session=session).final_output
            if current_agent == bjp_agent:
                print(f"{Fore.LIGHTRED_EX}{answer}{Fore.RESET}")
                current_agent = congress_agent
            else:
                print(f"{Fore.BLUE}{answer}{Fore.RESET}")
                current_agent = bjp_agent

##########################################
# Example: Dynamic Orchestration         #
##########################################
class DynamicOrchestrationExample:
    def run(self):
        bjp_agent = Agent(
            name="BJP Supporter",
            model=model,
            instructions="""
            You are a BJP supporter and you have to answer all questions in favor of BJP.
            Always start with 'As a BJP supporter,' in your answers.
            """,
            model_settings=ModelSettings(temperature=0.0, top_p=1.0)
        )
        congress_agent = Agent(
            name="Congress Supporter",
            model=model,
            instructions="""
            You are a Congress supporter and you have to answer all questions in favor of Congress/INC.
            Always start with 'As a Congress supporter,' in your answers.
            """,
            model_settings=ModelSettings(temperature=0.0, top_p=1.0)
        )
        current_agent = congress_agent
        session = SQLiteSession(
            db_path="../dynamic_memory.db",
            session_id="political_debate_dynamic")
        for _ in range(4):
            agent = Agent(
                name="Orchestrator Agent",
                model=model,
                tools=[bjp_agent.as_tool(tool_name="bjp_agent_tool",
                                          tool_description="Answers questions in favor of BJP."),
                       congress_agent.as_tool(tool_name="congress_agent_tool",
                                              tool_description="Answers questions in favor of Congress/INC.")],
                instructions="""
                You are an orchestrator agent that decides which political party agent to use based on the last response.
                If the last response was in favor of BJP, use the congress_agent_tool next and 
                if the last response was in favour of Congress use the bjp_agent_tool next.
                """,
                model_settings=ModelSettings(temperature=0.0, top_p=1.0)
            )
            print(Runner.run_sync(agent, """You have to have 6 conversations to and fro to justify 
                    why you are better for India and how the other party is bad. only provide one sentence at a time.
                    Always counter the previous point of the other party.""", session=session).final_output)
            sleep(1)

#################################################
# Example: Multi Agent Switching                #
#################################################
class MultiAgentSwitchingExample:
    def run(self):
        ##################from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX#########################
        # You are part of a multi-agent system called the Agents SDK, designed to make agent coordination and execution #
        # easy. Agents uses two primary abstraction: **Agents** and **Handoffs**. An agent encompasses instructions and #
        # tools and can hand off a conversation to another agent when appropriate. Handoffs are achieved by calling a   #
        # handoff function, generally named `transfer_to_<agent_name>`. Transfers between agents are handled seamlessly #
        # in the background; do not mention or draw attention to these transfers in your conversation with the user.    #
        #################################################################################################################
        print(f"{RECOMMENDED_PROMPT_PREFIX}")  # Also called Handoff Prompting
        complaints_agent = Agent(
            name="Complaints Agent",
            model=model,
            instructions=f"{RECOMMENDED_PROMPT_PREFIX} You are a customer complaints assistant for Netflix.. You handle customer complaints."
        )
        sales_agent = Agent(
            name="Sales Agent",
            model=model,
            instructions="You are a sales assistant for Netflix. You handle customer sales inquiries."
        )
        technical_support_agent = Agent(
            name="Technical Support Agent",
            model=model,
            instructions=f"{RECOMMENDED_PROMPT_PREFIX} You are a technical support assistant for Netflix. You handle customer technical support inquiries."
        )
        general_support_agent = Agent(
            name="General Support Agent",
            model=model,
            instructions=f"{RECOMMENDED_PROMPT_PREFIX} You are a general support assistant for Netflix. You handle general customer inquiries."
        )
        complaints_agent.handoffs = [general_support_agent, sales_agent]
        sales_agent.handoffs = [general_support_agent, complaints_agent]
        general_support_agent.handoffs = [complaints_agent, sales_agent, technical_support_agent]
        last_agent = general_support_agent
        session = SQLiteSession("first_session")
        for _ in range(5):
            question = input("You:")
            result = Runner.run_sync(last_agent, question, session=session)
            print(f"{Fore.GREEN}A: {result.final_output}{Fore.RESET}")
            #################################################################################
            # IMPORTANT!                                                                    #
            # When you call Runner.run_sync(), it returns a result object that contains:    #
            # final_output: The text response to show the user                              #
            # last_agent: A reference to whichever agent finished handling the request      #
            #################################################################################
            last_agent = result.last_agent

#####################################################
# Example: Hierarchical Multi-Agent System          #
#####################################################
class HierarchicalMultiAgentExample:
    def run(self):
        business_analyst_agent = Agent(
            name="Business Analyst Agent",
            model=model,
            instructions="You are a business analyst assistant. You gather requirements from stakeholders and communicate them to the project manager."
        )
        trainee_developer_agent = Agent(
            name="Trainee Developer Agent",
            model=model,
            instructions="You are a trainee developer assistant. You assist the developer in completing tasks assigned by the developer."
        )
        developer_agent = Agent(
            name="Developer Agent",
            model=model,
            instructions="You are a developer assistant. You complete tasks assigned by the team lead.",
            handoffs=[trainee_developer_agent]
        )
        tester_agent = Agent(
            name="Tester Agent",
            model=model,
            instructions="You are a tester assistant. You test the code developed by the developer."
        )
        development_team_lead_agent = Agent(
            name="Development Team Lead Agent",
            model=model,
            instructions="You are a Development team lead assistant. You manage tasks assigned by the project manager and delegate them to team members.",
            handoffs=[developer_agent]
        )
        testing_team_lead_agent = Agent(
            name="Testing Team Lead Agent",
            model=model,
            instructions="You are a Testing team lead assistant. You manage testing tasks assigned by the project manager and delegate them to team members.",
            handoffs=[tester_agent]
        )
        project_manager_agent = Agent(
            name="Project Manager Agent",
            model=model,
            instructions="You are a project manager assistant. You manage tasks and delegate them to appropriate team members.",
            handoffs=[business_analyst_agent, development_team_lead_agent, testing_team_lead_agent]
        )
        session = SQLiteSession("hierarchical_multi_agent_session")
        last_agent = project_manager_agent
        for _ in range(5):
            question = input("You:")
            result = Runner.run_sync(last_agent, question, session=session)
            print(f"{Fore.GREEN}A: {result.final_output}{Fore.RESET}")
            last_agent = result.last_agent


def main():
    examples = [
        # BasicExample,
        # PydanticExample,
        # FunctionToolExample,
        # HandOffExample,
        # GuardrailsExample
        # ToolUseExample
        # ToolStoppingExample,
        # FunctionChainingExample,
        # ShortTermMemoryExample,
        # PersistentMemoryExample,
        # DeterministicOrchestrationExample,
        # DynamicOrchestrationExample,
        # MultiAgentSwitchingExample,
        HierarchicalMultiAgentExample,
    ]
    for example in examples:
        print(f"Running example: {example.__name__}")
        example().run()
        sleep(2)
        print("-" * 40)

if __name__ == "__main__":
    main()