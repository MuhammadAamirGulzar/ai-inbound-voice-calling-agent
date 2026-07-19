if __name__ == "__main__":
    from base import BaseChatbot
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from Minibot import Minibot
    from constants import FILLERS
else:
    from .base import BaseChatbot
    from Minibot import Minibot
    from constants import FILLERS

import os
import json
import random
import re
from dotenv import load_dotenv
from openai import OpenAI
from openai._types import NOT_GIVEN

import send_email

def schedule_call(email):
    send_email(receiver=email, subject="Call Scheduled!", body="Your call has been scheduled with our agent.")
    return {"status": "Call scheduled", "email": email}


functions = [
    {
        "name": "schedule_call",
        "description": "Schedule a call when the user requests to speak to an agent or wants to discuss something that would require human involvement.",
        "parameters":{
            "email": {"type":"string","description":"The email address of the user."},
            "date": {"type":"datetime","description":"The date and time for the call."},
            "required": ["email"],
        }
    }
]


class Chatbot_LLM(BaseChatbot):
    def __init__(
        self,
        sys_prompt="",
        Model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key="",
        tools=None,
        tool_choice=NOT_GIVEN,
        tool_utterances=None,
        functions=None,
    ):
        if tools is None:
            tools = NOT_GIVEN
        if tool_utterances is None:
            tool_utterances = {}
        if functions is None:
            self.functions = {}
        if api_key == "":
            load_dotenv()
            api_key = os.getenv("LLM_EC2_KEY")
            

        self.MODEL = Model
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        #print("Client initialized:", self.client)  # Debugging line
        self.messages = []
        self.messages.append({"role": "system", "content": sys_prompt})
        self.tools = tools
        self.tool_choice = tool_choice
        self.tool_utterances = tool_utterances
        self.functions = functions

        self.minibot = Minibot(THRESHOLD=0.75)  # Probabilistic model for word fillers
        self.set_minibot_args = True

    def run(self, input_text, minibot_args=None):
        self.messages.append({"role": "user", "content": input_text})

        # if not self.set_minibot_args:
        #     self.minibot.fill_faqs(minibot_args)
        #     self.set_minibot_args = True

        # clean_input_text = re.sub(r"[^\w\s]", "", input_text).lower()
        # response, confidence_score = self.minibot.get_response(clean_input_text)
        # if (
        #     response != self.minibot.UNKNOWN
        #     and confidence_score >= self.minibot.THRESHOLD
        # ):
        #     print("Response from Probabilistic Model.")
        #     yield response
        # else:
        print("Fetching response from LLM.")
        #yield "Filler:" + random.choice(FILLERS) + " "
        finished = False
        while not finished:
            func_call = dict()
            function_call_detected = False


            

            stream = self.client.chat.completions.create(
                model=self.MODEL,
                messages=self.messages,
                stream=True,
                tools=self.tools,
                tool_choice=self.tool_choice,
            )

            for chunk in stream:
                finish_reason = chunk.choices[0].finish_reason
                if chunk.choices[0].delta.tool_calls is not None:
                    function_call_detected = True
                    tool_call = chunk.choices[0].delta.tool_calls[0]
                    if tool_call.function.name:
                        func_call["name"] = tool_call.function.name
                        func_call["id"] = tool_call.id
                        func_call["arguments"] = ""
                        # Choose a utterance for the tool at random and output it for the tts
                        yield random.choice(
                            self.tool_utterances[func_call["name"]]
                        ) + " . "  # The period is to make it say immediately
                    if tool_call.function.arguments:
                        func_call["arguments"] += tool_call.function.arguments
                if function_call_detected and finish_reason == "tool_calls":
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": func_call["id"],
                                    "type": "function",
                                    "function": {
                                        "name": func_call["name"],
                                        "arguments": func_call["arguments"],
                                    },
                                }
                            ],
                        }
                    )
                    # Run the function
                    function_response = self.functions[func_call["name"]](
                        **json.loads(func_call["arguments"])
                    )
                    self.messages.append(
                        {
                            "tool_call_id": func_call["id"],
                            "role": "tool",
                            "name": func_call["name"],
                            "content": function_response,
                        }
                    )
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                if finish_reason == "stop":
                    finished = True

    def post_process(self, response):
        # Remove the tool utterances from the response
        # for tool in self.tool_utterances:
        #     response = response.replace(self.tool_utterances[tool], '')
        self.messages.append({"role": "assistant", "content": response})
        return response


import speech_recognition as sr


def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            print("Recognizing...")
            user_input = recognizer.recognize_google(audio)
            print(f"User: {user_input}")
            return user_input
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
        except sr.RequestError:
            print("Could not request results from the STT service.")
        return None


if __name__ == "_main_":
    minibot_args = {
        "agent_name": "Alice",
        "organization_name": "Medicare",
        "total_products": 1,
        "product_names": ["Hospital Insurance"],
        "product_descriptions": [
            "Covers inpatient care in hospitals, skilled nursing facility care, hospice care, and some home health services."
        ],
        "product_features": [
            "Free for Citizens above 70"
        ],
    }

    chatbot = Chatbot_LLM()
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "stop"]:
            break
        response = chatbot.generate_response(user_input, minibot_args)
        print(f"Bot: {response}\n")