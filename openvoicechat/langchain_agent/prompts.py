from langchain.prompts import StringPromptTemplate
from typing import List

class InfoCollectionAgentPrompt(StringPromptTemplate):
    """Prompt template for the information collection agent."""
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action.tool}\nAction Input: {action.tool_input}\nObservation: {observation}\n"
        
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        
        # Return the formatted template
        return self.template.format(**kwargs)

def get_agent_prompt(agent_name: str, organization_name: str):
    """Returns the prompt template for the agent."""
    
    template = f"""You are {agent_name}, a friendly sales agent for {organization_name}. Your goal is to:

1. Introduce {organization_name} and its offerings
2. Collect and verify the following information from the user:
   - Name (confirm by spelling it out letter by letter)
   - Age
   - City
   - Zipcode (read back as individual digits, e.g., "four six zero zero six")
   - Interest in services

Current user information: {{user_info}}

Follow these guidelines:
- Be conversational but efficient
- After collecting each piece of information, verify it by asking "Is that correct?"
- For names, spell out each letter (e.g., "S-a-l-m-a-n, is that right?")
- For zipcodes, read back each digit individually
- If asked about services, explain them before returning to information collection
- After collecting all information, thank the user and explain next steps
- IMPORTANT: Never ask for information you already have

User message: {{input}}
{{agent_scratchpad}}

What would you like to say to the user?
"""
    
    return template