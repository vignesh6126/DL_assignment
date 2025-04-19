import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Dropout
from keras.optimizers import Adam

# ------------------------------
# 1. Load and preprocess the data
# ------------------------------

def load_data(file_path, num_samples=None):
    input_texts = []
    target_texts = []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    print(f"Number of lines in file: {len(lines)}")

    for line in lines[:num_samples]:
        if not line.strip():
            continue  # Skip empty lines

        print(f"Processing line: {line}")

        # Split the line into 3 parts based on tabs
        parts = line.split('\t')

        # Check if there are at least two parts (Devanagari and Latin transliteration)
        if len(parts) >= 2:
            input_text, target_text = parts[1], parts[0]  # Devanagari is the first part, Latin is the second part
            input_texts.append(input_text.lower())  # Convert to lowercase
            target_texts.append('\t' + target_text + '\n')  # Add start and end tokens
        else:
            print(f"Skipping invalid line (not enough parts): {line}")

    print(f"Loaded {len(input_texts)} samples from {file_path}")
    return input_texts, target_texts


train_input_texts, train_target_texts = load_data('hi.translit.sampled.train.tsv', num_samples=5000)
val_input_texts, val_target_texts = load_data('hi.translit.sampled.dev.tsv', num_samples=500)
test_input_texts, test_target_texts = load_data('hi.translit.sampled.test.tsv', num_samples=500)

# ------------------------------
# 2. Build character-level vocabulary
# ------------------------------

def build_vocab(texts):
    all_chars = set(''.join(texts))
    all_chars.add('<UNK>')  # Add an "unknown" token for missing characters
    char2idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
    idx2char = {idx: char for char, idx in char2idx.items()}
    return char2idx, idx2char

# Build vocab for target language
target_char2idx, target_idx2char = build_vocab(train_target_texts)
input_char2idx, input_idx2char = build_vocab(train_input_texts)

print(f"Target vocab size: {len(target_char2idx)}")
print(f"Sample characters in target vocab: {list(target_char2idx.keys())[:10]}")

# ------------------------------
# 3. Vectorization
# ------------------------------

max_encoder_seq_length = max(len(txt) for txt in train_input_texts)
max_decoder_seq_length = max(len(txt) for txt in train_target_texts)

def vectorize(input_texts, target_texts):
    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype='float32')
    decoder_input_data = np.zeros((len(target_texts), max_decoder_seq_length), dtype='float32')
    decoder_target_data = np.zeros((len(target_texts), max_decoder_seq_length, len(target_char2idx)), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t] = input_char2idx.get(char, input_char2idx['<UNK>'])  # Use <UNK> for missing characters
        for t, char in enumerate(target_text):
            decoder_input_data[i, t] = target_char2idx.get(char, target_char2idx['<UNK>'])  # Use <UNK> for missing characters
            if t > 0:
                decoder_target_data[i, t - 1, target_char2idx.get(char, target_char2idx['<UNK>'])] = 1.0

    return encoder_input_data, decoder_input_data, decoder_target_data

# Vectorizing data
encoder_input_data, decoder_input_data, decoder_target_data = vectorize(train_input_texts, train_target_texts)
val_encoder_input_data, val_decoder_input_data, val_decoder_target_data = vectorize(val_input_texts, val_target_texts)

# ------------------------------
# 4. Build the Seq2Seq Model
# ------------------------------

embedding_dim = 256  # Increased embedding size
hidden_dim = 512  # Increased hidden state size

num_encoder_tokens = len(input_char2idx)
num_decoder_tokens = len(target_char2idx)

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(num_encoder_tokens, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(hidden_dim, return_state=True, dropout=0.3, recurrent_dropout=0.3)  # Added dropout for regularization
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(num_decoder_tokens, embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True, dropout=0.3, recurrent_dropout=0.3)  # Added dropout for regularization
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Final model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------------
# 5. Train the Model
# ------------------------------

# Early stopping and model checkpointing
from keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('seq2seq_model.h5', save_best_only=True)  # Saving entire model

model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=64,
    epochs=50,
    validation_data=([val_encoder_input_data, val_decoder_input_data], val_decoder_target_data),
    callbacks=[early_stopping, model_checkpoint]
)


# ------------------------------
# 6. Inference Models for Decoding
# ------------------------------

# Encoder inference model
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder inference model
decoder_state_input_h = Input(shape=(hidden_dim,))
decoder_state_input_c = Input(shape=(hidden_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2
)

# ------------------------------
# 7. Decode sequences
# ------------------------------

def decode_sequence(input_seq):
    # Encode the input
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence with just the start character
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_char2idx['\t']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = target_idx2char[sampled_token_index]
        decoded_sentence += sampled_char

        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            break

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()

# ------------------------------
# 8. Evaluate on Test Data
# ------------------------------

test_encoder_input_data, _, _ = vectorize(test_input_texts, test_target_texts)

correct = 0
total = len(test_input_texts)

print("\nSample predictions:")
for i in range(10):  # Show 10 samples
    input_seq = test_encoder_input_data[i:i+1]
    decoded = decode_sequence(input_seq)
    actual = test_target_texts[i].strip('\t\n')
    source = test_input_texts[i]
    print(f"{source} → {decoded} (Actual: {actual})")
    if decoded == actual:
        correct += 1

accuracy = correct / total * 100
print(f"\n✅ Test Accuracy (exact match): {accuracy:.2f}%")
