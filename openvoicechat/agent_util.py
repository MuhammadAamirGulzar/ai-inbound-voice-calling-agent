import os
import queue
import random
import re
import csv
import threading
from openvoicechat.utils import log_to_file
from openvoicechat.langchain_agent.agent import InfoCollectionAgent
import time



def clean_text_for_tts(text):
    """
    Clean text from markdown and special characters before sending to TTS
    """
    # Remove markdown formatting
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold text
    cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)    # Italic text
    
    # Remove bullet points and numbering
    cleaned = re.sub(r'^\s*[\*\-•]\s+', '', cleaned, flags=re.MULTILINE)  # Bullet points
    cleaned = re.sub(r'^\s*\d+\.\s+', '', cleaned, flags=re.MULTILINE)    # Numbered lists
    
    # Replace special characters commonly found in markdown
    cleaned = cleaned.replace('**', '')
    cleaned = cleaned.replace('*', '')
    cleaned = cleaned.replace('`', '')
    cleaned = cleaned.replace('#', '')
    cleaned = cleaned.replace('>', '')
    
    return cleaned


class UserInformation:
    """Class to track and store user information during a conversation"""
    def __init__(self):
        self.name = None
        self.age = None
        self.city = None
        self.zipcode = None
        self.interest_level = None
        self.additional_notes = []
        self.collection_complete = False
        
        # Track verification status for each field
        self.verified = {
            'name': False,
            'age': False,
            'city': False,
            'zipcode': False,
            'interest': False
        }
        
        # Track current verification field if any
        self.current_verification = None
        
    def is_complete(self):
        """Check if all required information has been collected and verified"""
        all_collected = (self.name and self.age and self.city and 
                     self.zipcode and self.interest_level is not None)
        
        all_verified = all(self.verified.values())
        
        return all_collected and all_verified
    
    def get_next_question(self):
        """Return the next question to ask based on missing information or verification needs"""
        # If we're currently verifying something, continue with that
        if self.current_verification:
            return self.current_verification
            
        # Check if any collected information needs verification
        if self.name and not self.verified['name']:
            self.current_verification = "verify_name"
            return "verify_name"
        
        if self.age and not self.verified['age']:
            self.current_verification = "verify_age"
            return "verify_age"
            
        if self.city and not self.verified['city']:
            self.current_verification = "verify_city"
            return "verify_city"
            
        if self.zipcode and not self.verified['zipcode']:
            self.current_verification = "verify_zipcode"
            return "verify_zipcode"
            
        if self.interest_level and not self.verified['interest']:
            self.current_verification = "verify_interest"
            return "verify_interest"
        
        # If everything collected is verified, check what's missing
        if not self.name:
            return "get_name"
        elif not self.age:
            return "get_age"
        elif not self.city:
            return "get_city"
        elif not self.zipcode:
            return "get_zipcode"
        elif self.interest_level is None:
            return "get_interest"
        else:
            return "complete"
            
    def update_from_message(self, message, field=None):
        """Try to extract information from user message"""
        # First, check for verification responses (yes/correct/right/etc.)
        message_lower = message.lower()
        is_confirming = any(word in message_lower for word in 
                           ["yes", "correct", "right", "yeah", "yep", "sure", "exactly"])
        
        # If the user is confirming and we're verifying something
        if is_confirming and self.current_verification:
            if self.current_verification == "verify_name":
                self.verified['name'] = True
            elif self.current_verification == "verify_age":
                self.verified['age'] = True
            elif self.current_verification == "verify_city":
                self.verified['city'] = True
            elif self.current_verification == "verify_zipcode":
                self.verified['zipcode'] = True
            elif self.current_verification == "verify_interest":
                self.verified['interest'] = True
                
            # Clear the current verification now that it's done
            self.current_verification = None
            return
            
        # If we're in verification mode but user didn't confirm, keep current mode
        # but don't mark as verified yet
        if self.current_verification and not is_confirming:
            # If they're correcting information, try to extract again
            if field == "get_name" or self.current_verification == "verify_name":
                # Simple name extraction - first word that's capitalized
                words = message.split()
                for word in words:
                    if len(word) > 1 and word[0].isupper() and word.isalpha():
                        self.name = word
                        break
            
            # Continue with verification rather than proceeding to next field
            return
        
        # Regular information extraction
        if field == "get_name" or field is None:
            # Simple name extraction - first word that's capitalized
            words = message.split()
            for word in words:
                if len(word) > 1 and word[0].isupper() and word.isalpha():
                    self.name = word
                    break
        
        if field == "get_age" or field is None:
            # Extract numbers that could be age (between 18-99)
            import re
            age_matches = re.findall(r'\b([1-9][0-9])\b', message)
            for match in age_matches:
                age = int(match)
                if 18 <= age <= 99:
                    self.age = age
                    break
        
        if field == "get_city" or field is None:
            # This would need more sophisticated processing in a real system
            # For now, assume the city is mentioned after "in" or "from"
            words = message.lower().split()
            for i, word in enumerate(words):
                if word in ["in", "from"] and i < len(words) - 1:
                    # Take the next word if it's capitalized in the original message
                    original_words = message.split()
                    if i+1 < len(original_words) and original_words[i+1][0].isupper():
                        self.city = original_words[i+1]
                        # Remove any punctuation
                        self.city = self.city.rstrip(',.!?')
                        break
        
        if field == "get_zipcode" or field is None:
            # Look for 5-digit numbers that could be zipcodes
            import re
            zip_matches = re.findall(r'\b(\d{5})\b', message)
            if zip_matches:
                self.zipcode = zip_matches[0]
        
        if field == "get_interest" or field is None:
            # Check for interest indicators
            message_lower = message.lower()
            if any(word in message_lower for word in ["yes", "interested", "sure", "definitely", "absolutely"]):
                self.interest_level = "Interested"
            elif any(word in message_lower for word in ["no", "not interested", "nope", "pass"]):
                self.interest_level = "Not interested"
            elif any(word in message_lower for word in ["maybe", "perhaps", "not sure", "think about"]):
                self.interest_level = "Maybe interested"

    def spell_name(self):
        """Return the name spelled out letter by letter"""
        if not self.name:
            return ""
        return "-".join(self.name)
        
    def format_zipcode_for_speech(self):
        """Format zipcode for speech (e.g., 46006 becomes 'four six zero zero six')"""
        if not self.zipcode:
            return ""
            
        digit_words = {
            '0': 'zero',
            '1': 'one',
            '2': 'two',
            '3': 'three',
            '4': 'four',
            '5': 'five',
            '6': 'six',
            '7': 'seven',
            '8': 'eight',
            '9': 'nine'
        }
        
        return " ".join(digit_words[digit] for digit in str(self.zipcode))

    def __str__(self):
        """Return a string representation of the collected information"""
        info = []
        if self.name:
            info.append(f"Name: {self.name}")
        if self.age:
            info.append(f"Age: {self.age}")
        if self.city:
            info.append(f"City: {self.city}")
        if self.zipcode:
            info.append(f"Zipcode: {self.zipcode}")
        if self.interest_level:
            info.append(f"Interest: {self.interest_level}")
        if self.additional_notes:
            info.append(f"Notes: {'; '.join(self.additional_notes)}")
        return ", ".join(info)


def run_chat_agent(
    mouth,
    ear,
    chatbot,
    minibot_args=None,
    verbose=True,
    stopping_criteria=lambda x: False,
    starting_message=True,
    logging_path="chat_log.txt",
    save_path="user_info.csv",
    timing_callback=None
):
    """
    Enhanced chat function that uses LangChain agent to collect user information.
    Now with parallel sentence processing for faster responses.
    """
    print("Running INFO AGENT")
    
    # Initialize user information tracker
    user_info = UserInformation()
    
    def process_response_by_sentences(response):
        """Break response into sentences and process through TTS incrementally"""
        # Split response into sentences
        sentences = []
        for sep in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
            parts = response.split(sep)
            if len(parts) > 1:
                for i in range(len(parts) - 1):
                    sentences.append(parts[i] + sep.strip())
                # Last part may be a partial sentence
                if parts[-1].strip():
                    sentences.append(parts[-1].strip())
                break
        
        # If no sentence breaks, treat as one sentence
        if not sentences and response.strip():
            sentences = [response.strip()]
        
        # Process each sentence through TTS
        for sentence in sentences:
            if sentence.strip():
                cleaned_sentence = clean_text_for_tts(sentence.strip())
                mouth.say_text(cleaned_sentence)
    
    try:
        # Import here to avoid circular imports
        from openvoicechat.langchain_agent.agent import InfoCollectionAgent
        
        # Extract agent and organization info
        agent_name = minibot_args.get("agent_name", "Agent")
        organization_name = minibot_args.get("organization_name", "Our Company")
        
        # Get product information
        products = []
        if minibot_args:
            product_names = minibot_args.get("product_names", [])
            product_descriptions = minibot_args.get("product_descriptions", [])
            product_features = minibot_args.get("product_features", [])
            
            for i in range(len(product_names)):
                product = {
                    "name": product_names[i] if i < len(product_names) else "",
                    "description": product_descriptions[i] if i < len(product_descriptions) else "",
                    "feature": product_features[i] if i < len(product_features) else ""
                }
                products.append(product)
        
        # Create LangChain agent
        agent = InfoCollectionAgent(
            user_info_instance=user_info,
            agent_name=agent_name,
            organization_name=organization_name,
            products=products
        )
        
        # Initial message
        if starting_message:
            product_desc = ""
            if minibot_args and "product_names" in minibot_args and minibot_args["product_names"]:
                product_desc = f", offering {minibot_args['product_names'][0]}"
                
            initial_message = f"Hello! I'm {agent_name} from {organization_name}{product_desc}. I'd like to assist you today by gathering some basic information. May I start by asking your name?"
            mouth.say_text(initial_message)
        
        # Main conversation loop
        pre_interruption_text = ""
        while True:
            # Start timing the total cycle
            total_cycle_start = time.time()
            
            # Time STT
            stt_start = time.time()
            user_input = pre_interruption_text + " " + ear.listen()
            stt_end = time.time()
            if timing_callback:
                timing_callback("stt", stt_start, stt_end)
            
            if verbose:
                print("USER: ", user_input)
            if os.environ.get("LOGGING", 0):
                log_to_file(logging_path, "USER: " + user_input)
            
            # Process the message using the LangChain agent
            # Time LLM
            llm_start = time.time()
            response = agent.process_message(user_input) 
            llm_end = time.time()
            if timing_callback:
                timing_callback("llm", llm_start, llm_end)
            
            # Update chatbot context to maintain state
            chatbot.messages.append({"role": "user", "content": user_input})
            chatbot.messages.append({"role": "assistant", "content": response})
            
            # Time TTS (for the whole process)
            tts_start = time.time()
            
            # Process by sentences for faster response
            process_response_by_sentences(response)
            
            tts_end = time.time()
            if timing_callback:
                timing_callback("tts", tts_start, tts_end)
            
            # Record the total cycle time
            total_cycle_end = time.time()
            if timing_callback:
                timing_callback("total_cycle", total_cycle_start, total_cycle_end)
            
            if verbose:
                print("BOT: ", response)
            if os.environ.get("LOGGING", 0):
                log_to_file(logging_path, "BOT: " + response)
            
            # Check if all information has been collected
            if user_info.is_complete() and not user_info.collection_complete:
                user_info.collection_complete = True
                
                # Save information to CSV
                is_new_file = not os.path.exists(save_path)
                with open(save_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    if is_new_file:
                        writer.writerow(['Name', 'Age', 'City', 'Zipcode', 'Interest'])
                    writer.writerow([
                        user_info.name, 
                        user_info.age, 
                        user_info.city, 
                        user_info.zipcode, 
                        user_info.interest_level
                    ])
                
                if verbose:
                    print("INFO COLLECTED: ", user_info)
                if os.environ.get("LOGGING", 0):
                    log_to_file(logging_path, "INFO COLLECTED: " + str(user_info))
                
                # End the conversation after saving data
                break
                
    except Exception as e:
        print(f"Error in LangChain agent: {e}")
        print("Falling back to standard agent...")