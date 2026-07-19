import re
from typing import Any, Dict, List


def clean_test(text: str = "") -> str:
    text = re.sub(r"\.\.", ".", text.lower())
    return text


def get_faqs_dict(args: Dict[str, Any] = None) -> Dict[str, List[str]]:
    for key, value in args.items():
        if isinstance(value, str):
            args[key] = clean_test(value)

    agent_name: str = args["agent_name"]
    organization_name: str = args["organization_name"]
    total_products: int = args["total_products"]
    product_names_list: List[str] = args["product_names"]
    product_names_list = [clean_test(product) for product in product_names_list]
    product_names_str: str = " and ".join(product_names_list)
    product_descriptions = args["product_descriptions"]
    product_features = args["product_features"]

    PRODUCT_QUESTIONS = [
        f"can you tell me more about the {{}}",
        f"do you have more information regarding the {{}}",
        f"i want to know more about the {{}}",
        f"what is the {{}}",
    ]

    FAQS_DICT = {
        # Others: Organization
        f"can you tell me about {organization_name}": ["*"],
        f"can you tell me more about {organization_name}": ["*"],
        # Greetings
        "hello": [
            "Greetings! How can I help?",
            "Hello!",
            "Hello! How can I assist you today?",
            "Hi! How can I help?",
            "Hi there! What can I do for you?",
        ],
        "hey": [
            "Greetings! How can I help?",
            "Hello!",
            "Hello! How can I assist you today?",
            "Hi! How can I help?",
            "Hi there! What can I do for you?",
        ],
        "hey there": [
            "Greetings! How can I help?",
            "Hello!",
            "Hello! How can I assist you today?",
            "Hi! How can I help?",
            "Hi there! What can I do for you?",
        ],
        "hi": [
            "Greetings! How can I help?",
            "Hello!",
            "Hello! How can I assist you today?",
            "Hi! How can I help?",
            "Hi there! What can I do for you?",
        ],
        # Farewell
        "bye": [
            "Goodbye!",
            "Goodbye! Have a great day!",
            "Goodbye! See you later!",
            "Goodbye! Talk to you soon!",
            "See you! Have a great day!",
        ],
        "good bye": [
            "Goodbye!",
            "Goodbye! Have a great day!",
            "Goodbye! See you later!",
            "Goodbye! Talk to you soon!",
            "See you! Have a great day!",
        ],
        "later": [
            "Goodbye!",
            "Goodbye! Have a great day!",
            "Goodbye! See you later!",
            "Goodbye! Talk to you soon!",
            "See you! Have a great day!",
        ],
        "see you": [
            "Goodbye!",
            "Goodbye! Have a great day!",
            "Goodbye! See you later!",
            "Goodbye! Talk to you soon!",
            "See you! Have a great day!",
        ],
        "see you later": [
            "Goodbye!",
            "Goodbye! Have a great day!",
            "Goodbye! See you later!",
            "Goodbye! Talk to you soon!",
            "See you! Have a great day!",
        ],
        "see you soon": [
            "Goodbye!",
            "Goodbye! Have a great day!",
            "Goodbye! See you later!",
            "Goodbye! Talk to you soon!",
            "See you! Have a great day!",
        ],
        # Thanking
        "thank you": [
            "No problem!",
            "No worries! I'm here to help.",
            "You're welcome!",
            "I'm glad I could help!",
            "Anytime! Feel free to ask if you need more help.",
        ],
        "thank you for your help": [
            "No problem!",
            "No worries! I'm here to help.",
            "You're welcome!",
            "I'm glad I could help!",
            "Anytime! Feel free to ask if you need more help.",
        ],
        "thank you so much": [
            "No problem!",
            "No worries! I'm here to help.",
            "You're welcome!",
            "I'm glad I could help!",
            "Anytime! Feel free to ask if you need more help.",
        ],
        "thanks": [
            "No problem!",
            "No worries! I'm here to help.",
            "You're welcome!",
            "I'm glad I could help!",
            "Anytime! Feel free to ask if you need more help.",
        ],
        "thanks a lot": [
            "No problem!",
            "No worries! I'm here to help.",
            "You're welcome!",
            "I'm glad I could help!",
            "Anytime! Feel free to ask if you need more help.",
        ],
        # Demographics
        "who are you": [
            f"I'm {agent_name} from {organization_name}.",
            f"My name is {agent_name}.",
            f"You can call me {agent_name}.",
            f"You're speaking with {agent_name}, from {organization_name}.",
        ],
        "whats your name": [
            f"I'm {agent_name} from {organization_name}.",
            f"My name is {agent_name}.",
            f"You can call me {agent_name}.",
            f"You're speaking with {agent_name}, from {organization_name}.",
        ],
        "who am i talking to": [
            f"I'm {agent_name} from {organization_name}.",
            f"My name is {agent_name}.",
            f"You can call me {agent_name}.",
            f"You're speaking with {agent_name}, from {organization_name}.",
        ],
        # Products and/or Services
        f"can you tell me about {organization_name} offerings": [
            f"{organization_name} has {total_products} main products {product_names_str}.",
            f"{organization_name} offers {total_products} main products {product_names_str}.",
            f"{organization_name} provides {product_names_str}.",
            f"We have {total_products} products {product_names_str}.",
        ],
        f"can you tell me about the products of {organization_name}": [
            f"{organization_name} has {total_products} main products {product_names_str}.",
            f"{organization_name} offers {total_products} main products {product_names_str}.",
            f"{organization_name} provides {product_names_str}.",
            f"We have {total_products} products {product_names_str}.",
        ],
        f"what products does {organization_name} offer": [
            f"{organization_name} has {total_products} main products {product_names_str}.",
            f"{organization_name} offers {total_products} main products {product_names_str}.",
            f"{organization_name} provides {product_names_str}.",
            f"We have {total_products} products {product_names_str}.",
        ],
        f"what does {organization_name} offer": [
            f"{organization_name} has {total_products} main products {product_names_str}.",
            f"{organization_name} offers {total_products} main products {product_names_str}.",
            f"{organization_name} provides {product_names_str}.",
            f"We have {total_products} products {product_names_str}.",
        ],
    }

    for index, product in enumerate(product_names_list):
        for question in PRODUCT_QUESTIONS:
            key = question.format(product)
            FAQS_DICT[key] = ["*"]

    return FAQS_DICT


# FILLERS = [
#     "Absolutely, let's get into that.",
#     "Certainly, that's worth looking into.",
#     "Give me a moment to double-check that.",
#     "Give me just a moment to get that information.",
#     "Good point, let's go over it.",
#     "Good question! Let me check for you.",
#     "Got it, let's dive into the details.Hmm, let me think about that.",
#     "Great question, let me explain.",
#     "Hmm, let me give that some thought.",
#     "Hmm, let me pull that up for you.",
#     "Hmm, let me see.",
#     "I appreciate your patience, just a second.",
#     "I'm thinking… let me get back to you in a second.",
#     "Interesting, give me a moment to consider it.",
#     "Let me find the best answer for you.",
#     "Let me make sure I have the right details for you.",
#     "Let me think for a second.",
#     "Let me think that through for a moment.",
#     "Let me verify that for you, hang tight.",
#     "Makes sense, let me check.",
#     "Of course! Let me confirm.",
#     "Of course, just a moment.",
#     "Right, let me gather the details for you.",
#     "Sure, let me clarify that.",
#     "Hmm, give me a moment.",
# ]

# FILLERS = [
#     "Absolutely, let's get into that.",
#     "Umm, okay got it, so…",
#     "Umm right, I'll tell you about that, so….",
#     "Hmm, let me think about that.",
#     "Hmm, let me give that some thought.",
#     "Hmm, let me pull that up for you.",
#     "Of course, just a moment.",
#     "Right, let me gather the details for you.",
#     "Sure, let me clarify that.",
#     "Cool, let me just look into that quickly.",
#     "Let me take a moment to gather that info.",
#     "Okay, let me get that for you right away.",
# ]

FILLERS = [
    "Hmm.....",
    "Well......",
    "Ah…..",
    "Okay...",
    "Right....",
    "I see....",
    "Got it....",
    "Alright.....",
    "Just a moment....",
]

ELEVENLABS_VOICE_IDS = {
    "Adam (Legacy)": "pNInz6obpgDQGcFmaJgB",
    "Alice": "Xb7hH8MSUJpSbSDYk0k2",
    "Antoni (Legacy)": "ErXwobaYiN019PkySvjV",
    "Arnold (Legacy)": "VR6AewLTigWG4xSOukaG",
    "Bill": "pqHfZKP75CvOlQylNhV4",
    "Brian": "nPczCjzI2devNBz1zQrb",
    "Callum": "N2lVS1w4EtoT3dr4eOWO",
    "Charlie": "IKne3meq5aSn9XLyUdCD",
    "Charlotte": "XB0fDUnXU5powFXDhCwa",
    "Chris": "iP95p4xoKVk53GoZ742B",
    "Clyde (Legacy)": "2EiwWnXFnvU5JabPnv8n",
    "Daniel": "onwK4e9ZLuTAKqWW03F9",
    "Dave (Legacy)": "CYw3kZ02Hs0563khs1Fj",
    "Domi (Legacy)": "AZnzlk1XvdvUeBnXmlld",
    "Dorothy (Legacy)": "ThT5KcBeYPX3keUQqHPh",
    "Drew (Legacy)": "29vD33N1CtxCmqQRPOHJ",
    "Elli (Legacy)": "MF3mGyEYCl7XYWbV9V6O",
    "Emily (Legacy)": "LcfcDJNUP1GQjkzn1xUU",
    "Eric": "cjVigY5qzO86Huf0OWal",
    "Ethan (Legacy)": "g5CIjZEefAph4nQFvHAz",
    "Fin (Legacy)": "D38z5RcWu1voky8WS1ja",
    "Freya (Legacy)": "jsCqWAovK2LkecY7zXl4",
    "George": "JBFqnCBsd6RMkjVDRZzb",
    "Gigi (Legacy)": "jBpfuIE2acCO8z3wKNLl",
    "Giovanni (Legacy)": "zcAOhNBS3c14rBihAFp1",
    "Glinda (Legacy)": "z9fAnlkpzviPz146aGWa",
    "Grace (Legacy)": "oWAxZDx7w5VEj9dCyTzz",
    "Harry (Legacy)": "SOYHLrjzK2X1ezoPC6cr",
    "James (Legacy)": "ZQe5CZNOzWyzPSCn5a3c",
    "Jeremy (Legacy)": "bVMeCyTHy58xNoL34h3p",
    "Jessica": "cgSgspJ2msm6clMCkdW9",
    "Jessie (Legacy)": "t0jbNlBVZ17f02VDIeMI",
    "Joseph (Legacy)": "Zlb1dXrM653N07WRdFW3",
    "Josh (Legacy)": "TxGEqnHWrfWFTfGW9XjX",
    "Laura": "FGY2WhTYpPnrIDTdsKH5",
    "Liam": "TX3LPaxmHKxFdv7VOQHJ",
    "Lily": "pFZP5JQG7iQjIQuC4Bku",
    "Matilda": "XrExE9yKIg1WjnnlVkGX",
    "Michael (Legacy)": "flq6f7yk4E4fJM5XTYuZ",
    "Mimi (Legacy)": "zrHiDhphv9ZnVXBqCLjz",
    "Nicole (Legacy)": "piTKgcLEGmPE4e6mEKli",
    "Patrick (Legacy)": "ODq5zmih8GrVes37Dizd",
    "Paul (Legacy)": "5Q0t7uMcjvnagumLfvZi",
    "Rachel (Legacy)": "21m00Tcm4TlvDq8ikWAM",
    "Sam (Legacy)": "yoZ06aMxZJJ28mfd3POQ",
    "Sarah": "EXAVITQu4vr4xnSDxMaL",
    "Serena (Legacy)": "pMsXgVXv3BLzUgSXRplE",
    "Thomas (Legacy)": "GBv7mTt0atIp3Br8iCZE",
    "Will": "bIHbv24MWmeRgasZH58o",
}


TEST_QUESTIONS = [
    # Organization
    "Can you tell me about VECTOR?",
    "I want to know more about VECTOR.",
    "What does VECTOR do?",
    # Product and Services: 1
    "Can you tell me more about the Smart Mirror?",
    "Do you have more information regarding the Smart Mirror?",
    "I want to know more about the Smart Mirror.",
    "What is the Smart Mirror?",
    # Product and Services: 2
    "Can you tell me more about the Virtual Try-on Application?",
    "Do you have more information regarding the Virtual Try-on Application?",
    "I want to know more about the Virtual Try-on Application.",
    "What is the Virtual Try-on Application?",
    # To Be Added
    # "Tell me about your products",
    # "What products do you sell?",
    # "What services does VECTOR offer?",
    # "What can I expect from the Smart Mirror?",
    # "What can I expect from the Virtual Try-on Application?",
]
