import random
import json
import torch
from model import NeuralNet
from main import bag_of_words, tokenize
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

with open('intents_administrator.json', 'r', encoding='utf-8') as f:
    intents_admin = json.load(f)

with open('intents_student.json', 'r', encoding='utf-8') as f:
    intents_student = json.load(f)

FILE = 'data.pth'
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)

model.eval()

def chat_response(sentence):

    if sentence == 'Salir':
        return "Hasta luego"
    
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() > 0.90:
        for intent in intents['intents']:
            if tag == intent['tag']:
                response = random.choice(intent['responses'])
                return (f"{response}", tag)
            else: 
                return (f"Disculpa, pero no entendí...", tag)
    else:
        return (f"Disculpa, pero no entendí...", tag)

def chat_response_student(sentence):

    if sentence == 'Salir':
        return "Hasta luego"
    
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() > 0.90:
        for intent in intents_student['intents']:
            if tag == intent['tag']:
                response = random.choice(intent['responses'])
                return (f"{response}", tag)
            else: 
                return (f"Disculpa, pero no entendí...", tag)
    else:
        return (f"Disculpa, pero no entendí...", tag)
    
def chat_response_admin(sentence):

    if sentence == 'Salir':
        return "Hasta luego"
    
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() > 0.90:
        for intent in intents_admin['intents']:
            if tag == intent['tag']:
                response = random.choice(intent['responses'])
                return (f"{response}", tag)
            else: 
                return (f"Disculpa, pero no entendí...", tag)
    else:
        return (f"Disculpa, pero no entendí...", tag)

@app.route('/api/chat', methods=['POST'])
@cross_origin()
def chat():
    data = request.get_json()
    sentence = data.get('sentence', '')

    response,  tag = chat_response(sentence)
    return jsonify({'response': response, 'tag': tag})

@app.route('/api/chat/student', methods=['POST'])
@cross_origin()
def chat_student():
    data = request.get_json()
    sentence = data.get('sentence', '')

    response,  tag = chat_response_student(sentence)
    return jsonify({'response': response, 'tag': tag})

@app.route('/api/chat/admin', methods=['POST'])
@cross_origin()
def chat_admin():
    data = request.get_json()
    sentence = data.get('sentence', '')

    response, tag = chat_response_admin(sentence)
    return jsonify({'response': response, 'tag': tag})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int("5000"), debug=False)