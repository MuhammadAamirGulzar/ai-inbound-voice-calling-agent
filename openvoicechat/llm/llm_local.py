import queue
import requests
import json
import random
from .base import BaseChatbot  # or wherever your BaseChatbot is

class Chatbot_local(BaseChatbot):
    def __init__(
        self,
        endpoint_url="http://0.0.0.0:1234/v1/chat/completions",
        model="gemma-2-2b-it:2",
        sys_prompt="",
        temperature=0.7,
        max_tokens=-1,
    ):
        super().__init__()
        self.endpoint_url = endpoint_url
        self.model = model
        self.sys_prompt = sys_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.messages = []
        if sys_prompt:
            self.messages.append({"role": "system", "content": sys_prompt})

    def run(self, input_text: str, minibot_args=None):
        """
        Yields text chunks. Called by generate_response() or generate_response_stream().
        """
        # 1. Add user message to local conversation
        self.messages.append({"role": "user", "content": input_text})

        # 2. Quick "thinking" filler
        yield "Filler: " + random.choice(["Hmm...", "Let’s see...", "Alright..."]) + " "

        # 3. Prepare request payload
        payload = {
            "model": self.model,
            "messages": self.messages,      # system + history + user
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # 4. Call the local LLM with streaming enabled
        try:
            with requests.post(
                self.endpoint_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                stream=True,          # Important: streaming
                timeout=120
            ) as response:
                if response.status_code != 200:
                    yield f"Error calling local LLM: {response.status_code}, {response.text}"
                    return

                # 5. Read line by line from the response
                for line in response.iter_lines():
                    if not line:
                        continue
                    # The local LLM typically returns JSON lines
                    # Something like: data: {...json...}
                    decoded_line = line.decode('utf-8')
                    # If you see 'data: {...}', remove the 'data: '
                    if decoded_line.startswith("data: "):
                        decoded_line = decoded_line.replace("data: ", "")
                    if decoded_line == "[DONE]":
                        break

                    # 6. Parse the JSON
                    try:
                        data_json = json.loads(decoded_line)
                        if "choices" in data_json and len(data_json["choices"]) > 0:
                            delta = data_json["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        # If we fail to parse, just yield the raw line
                        yield decoded_line

        except requests.exceptions.RequestException as e:
            yield f"Error: {e}"

    def post_process(self, response):
        """
        Add the final assistant message to the conversation history
        and return the final text.
        """
        self.messages.append({"role": "assistant", "content": response})
        return response
