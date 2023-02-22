from revChatGPT.V1 import Chatbot
import IPython # Only for Google Colab
import time
from IPython.display import display

chatgpt_email = "leonhartdreyse@gmail.com"
chatgpt_pw = "gexvyf-witki1-duzfeK"

# def print_gpt(response, sleep = 0.1):
#     output = display('ChatGPT...',display_id=True)
#     for data in response:
#         output.update(IPython.display.Pretty(data['message']))
#         time.sleep(sleep)

def init_gpt(email,pw):
    return Chatbot(config={
        "email":email,
        "password":pw
    })
chatbot = init_gpt(chatgpt_email,chatgpt_pw)