from dotenv import load_dotenv
from config import Config

cfg = Config()

from llm_utils import create_chat_completion


def create_chat_message(role, content):
    """
    Create a chat message with the given role and content.

    Args:
    role (str): The role of the message sender, e.g., "system", "user", or "assistant".
    content (str): The content of the message.

    Returns:
    dict: A dictionary containing the role and content of the message.
    """
    return {"role": role, "content": content}



# TODO: Change debug from hardcode to argument
def chat_with_ai(
        prompt,
        user_input,
        full_message_history,
        permanent_memory,
        token_limit,
        debug=False):
    if cfg.openai_api_key:
        import chat_openai
        chat_openai.chat_with_openai(prompt,user_input,full_message_history,permanent_memory,token_limit,debug)
    elif cfg.llama_model_path:
        import chat_llama
        chat_llama.chat_with_llama(prompt,user_input,full_message_history,permanent_memory,token_limit,debug)
