import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load trained model
model = load_model("model_for_next_word.h5")


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

input_text = "To be or not to be"
max_sequence_len = model.input_shape[1] + 1
next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)

print(f"Input text: {input_text}")
print(f"Next Word Prediction: {next_word}")
