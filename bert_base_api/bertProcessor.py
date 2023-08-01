import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from flask import Flask, request, jsonify

# initate the App
app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertModel.from_pretrained(model_name)

@app.route('/embeddings', methods=['POST'])
def get_embeddings():
    data = request.json
    text = ""
    try:
        text = data['text']
        inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True)
        outputs = model(inputs)
        # Extract the embedding for the [CLS] token (first token)
        embedding = outputs.last_hidden_state[:, 0, :].numpy().tolist()
        return jsonify({"embedding": embedding})
    except KeyError:
        print(KeyError)
        return jsonify({"embedding": []})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002)
