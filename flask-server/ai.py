from flask import jsonify
import openai
import os
import sys

openai.api_key = os.getenv("OPEN_API_KEY")

def chatcompletion(user_input,chat_history):
  output = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=1,
    presence_penalty=0,
    frequency_penalty=0,
    messages=[
      {"role":"system", "content": "You are a math genius that can explain any math problem with simple steps."},
      {"role":"system", "content": "Check if the prompt given is a valid math problem. If the prompt is not a valid math problem, then only reply with 'Sorry, this is a invalid question'. This is very important"},
      {"role":"system", "content": "Write down the type of problem that is being solved as well."},
      {"role": "system", "content": f"Conversation history: {chat_history}"},
      {"role": "user", "content": f"{user_input}."},
    ]
  )

  for item in output['choices']:
    chatgpt_output = item['message']['content']

  return chatgpt_output

def ai(data):

    if(not any(str.isdigit(c) for c in data["message"])):
        return "Sorry, this is a invalid question. How can I help you with math?"

    chat_history = data["history"]

    user_input = data["message"]

    chatgpt_raw_output = chatcompletion(user_input, chat_history)

    return chatgpt_raw_output
