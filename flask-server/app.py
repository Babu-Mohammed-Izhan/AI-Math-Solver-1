from flask import Flask, request, jsonify
import openai
from flask_cors import CORS, cross_origin
import json

openai.api_key = "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"

app = Flask(__name__)

CORS(app, support_credentials=True)

def chatcompletion(user_input,chat_history):
  output = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    temperature=1,
    presence_penalty=0,
    frequency_penalty=0,
    messages=[
      {"role": "system", "content": f"Conversation history: {chat_history}"},
      {"role": "user", "content": f"{user_input}."},
    ]
  )

  for item in output['choices']:
    chatgpt_output = item['message']['content']

  return chatgpt_output

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        data = json.loads(request.data)

        if(not any(str.isdigit(c) for c in data["message"])):
           return jsonify({"botResponse": "Sorry, this is a invalid question" })

        chat_history = data["history"]

        user_input = data["message"]

        chatgpt_raw_output = chatcompletion(user_input, chat_history)


        return jsonify({"botResponse":chatgpt_raw_output})
    if request.method == "GET":
       print(request.method)
    return "This is the server"


if __name__ == '__main__':
    app.run(debug=True)
