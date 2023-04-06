import time
import traceback
import token_counter
from config import Config
from dotenv import load_dotenv
from llama_cpp import Llama

cfg = Config()
llm = Llama(model_path='/models/7b/model.bin',n_batch=6000)

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


def tokenize_chat_ctx(messages):
    """
    Create a chat message with the given role and content.

    Args:
    list (dict( str )): List of dictionary items that contain role, content, and optionally user

    Returns:
    int: number of tokens.
    """
    instructions = """Complete the following chat conversation between the user and the assistant. System messages should be strictly followed as additional instructions."""
    chat_history = "\n".join(
        f'{message["role"]} {message.get("user", "")}: {message["content"]}'
        for message in messages
    )
    prompt = f" \n\n### Instructions:{instructions}\n\n### Inputs:{chat_history}\n\n### Response:\nassistant: "
    return len(llm.tokenize(prompt.encode('utf-8')))

def chat_with_llama(
        prompt,
        user_input,
        full_message_history,
        permanent_memory,
        token_limit,
        debug=True):
    while True:
        try:
            """
            Interact with the OpenAI API, sending the prompt, user input, message history, and permanent memory.

            Args:
            prompt (str): The prompt explaining the rules to the AI.
            user_input (str): The input from the user.
            full_message_history (list): The list of all messages sent between the user and the AI.
            permanent_memory (list): The list of items in the AI's permanent memory.
            token_limit (int): The maximum number of tokens allowed in the API call.

            Returns:
            str: The AI's response.
            """
            current_context = [
                create_chat_message(
                    "system", prompt), create_chat_message(
                    "system", f"Permanent memory: {permanent_memory}")]
            
            user_input = [create_chat_message("user", user_input)]

            #if debug:
            print(f"Token limit: {token_limit}")
            print(str(tokenize_chat_ctx([create_chat_message("user", "test test test")])))
            # Count the currently used tokens plus the user input tokens for counting
            # Reserve 1000 tokens for the response

            send_token_limit = 2048 - 100 - tokenize_chat_ctx(current_context) - tokenize_chat_ctx(user_input)-1040

            # Start appending messages to current context until we near the limit
            new_messages = []
            new_message_tokens = 0
            #Go through the most recent messages
            for message in list(reversed(full_message_history)):
                new_message_tokens += len(llm.tokenize([message]))
                if new_message_tokens > send_token_limit:
                    break
                
                new_messages.insert(0, message)

            print("Chose tokens")
            # Add the most recent message to the start of the current context, after the two system prompts created above
            current_context = current_context + new_messages + user_input
            # Append user input, the length of this is accounted for above
            #current_context.append(user_input)

            # Calculate remaining tokens
            tokens_remaining = token_limit - tokenize_chat_ctx(current_context)

            # Debug print the current context
            #if debug:
            print(f"Token limit: {token_limit}")
            print(f"Send Token Count: {token_limit-tokens_remaining}")
            print(f"Tokens remaining for response: {tokens_remaining}")
            print("------------ CONTEXT SENT TO AI ---------------")
            for message in current_context:
                # Skip printing the prompt
                if message["role"] == "system" and message["content"] == prompt:
                    continue
                print(
                    f"{message['role'].capitalize()}: {message['content']}")
                print()
            print("----------- END OF CONTEXT ----------------")

            assistant_reply = create_chat_completion(
                messages=current_context,
                max_tokens=tokens_remaining,
            ).choices[0].message["content"]

            # Update full message history
            full_message_history.append(user_input)
            full_message_history.append(
                create_chat_message(
                    "assistant", assistant_reply))

            return assistant_reply
        except Exception as ex:
            # TODO: WHen we switch to langchain, this is built in
            print("Error: ", "Shits broke")
            print(ex)
            print(traceback.format_exc())
            time.sleep(1)

# Overly simple abstraction until we create something better
def create_chat_completion(messages, temperature=None, max_tokens=None)->str:
    response = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens)
    return response