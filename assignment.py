import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Dropout, Concatenate, Activation, Dot, Lambda
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ------------------------------
# 1. Data Loading and Preprocessing
# ------------------------------

def load_data(file_path, num_samples=None):
    input_texts = []
    target_texts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    for line in lines[:num_samples]:
        if not line.strip():
            continue
            
        parts = line.split('\t')
        if len(parts) >= 2:
            input_text = parts[1].strip().lower()
            target_text = parts[0].strip()
            
            if input_text and target_text:
                input_texts.append(input_text)
                target_texts.append('\t' + target_text + '\n')
    
    print(f"Loaded {len(input_texts)} samples from {file_path}")
    return input_texts, target_texts

train_input_texts, train_target_texts = load_data('hi.translit.sampled.train.tsv', num_samples=20000)
val_input_texts, val_target_texts = load_data('hi.translit.sampled.dev.tsv', num_samples=1000)
test_input_texts, test_target_texts = load_data('hi.translit.sampled.test.tsv', num_samples=1000)

# ------------------------------
# 2. Vocabulary Building
# ------------------------------

def build_vocab(texts, min_count=2):
    char_counts = {}
    for text in texts:
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
    
    all_chars = [char for char, count in char_counts.items() if count >= min_count]
    all_chars = sorted(all_chars)
    all_chars.extend(['<UNK>', '<PAD>'])
    
    char2idx = {char: idx for idx, char in enumerate(all_chars)}
    idx2char = {idx: char for char, idx in char2idx.items()}
    return char2idx, idx2char

target_char2idx, target_idx2char = build_vocab(train_target_texts)
input_char2idx, input_idx2char = build_vocab(train_input_texts)

print(f"Input vocab size: {len(input_char2idx)}")
print(f"Target vocab size: {len(target_char2idx)}")

# ------------------------------
# 3. Vectorization with Padding
# ------------------------------

max_encoder_seq_length = max(len(txt) for txt in train_input_texts) + 2
max_decoder_seq_length = max(len(txt) for txt in train_target_texts) + 2

def vectorize(input_texts, target_texts):
    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype='float32')
    decoder_input_data = np.zeros((len(target_texts), max_decoder_seq_length), dtype='float32')
    decoder_target_data = np.zeros((len(target_texts), max_decoder_seq_length, len(target_char2idx)), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t] = input_char2idx.get(char, input_char2idx['<UNK>'])
        for t in range(len(input_text), max_encoder_seq_length):
            encoder_input_data[i, t] = input_char2idx['<PAD>']
            
        for t, char in enumerate(target_text):
            decoder_input_data[i, t] = target_char2idx.get(char, target_char2idx['<UNK>'])
            if t > 0:
                decoder_target_data[i, t-1, target_char2idx.get(char, target_char2idx['<UNK>'])] = 1.0
        for t in range(len(target_text), max_decoder_seq_length):
            decoder_input_data[i, t] = target_char2idx['<PAD>']
            if t > 0:
                decoder_target_data[i, t-1, target_char2idx['<PAD>']] = 1.0
    
    return encoder_input_data, decoder_input_data, decoder_target_data

encoder_input_data, decoder_input_data, decoder_target_data = vectorize(train_input_texts, train_target_texts)
val_encoder_input_data, val_decoder_input_data, val_decoder_target_data = vectorize(val_input_texts, val_target_texts)
test_encoder_input_data, _, _ = vectorize(test_input_texts, test_target_texts)

# ------------------------------
# 4. Model with Custom Attention
# ------------------------------

embedding_dim = 256
hidden_dim = 512
def attention_layer(decoder_hidden, encoder_output):
    scores = Dot(axes=[2, 2])([decoder_hidden, encoder_output])
    attention_weights = Activation('softmax')(scores)
    context = Dot(axes=[2, 1])([attention_weights, encoder_output])
    return context

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(len(input_char2idx), embedding_dim, mask_zero=True)(encoder_inputs)
encoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True, 
                   dropout=0.4, recurrent_dropout=0.4)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb = Embedding(len(target_char2idx), embedding_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True,
                   dropout=0.4, recurrent_dropout=0.4)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
attention_context = attention_layer(decoder_outputs, encoder_outputs)
decoder_concat = Concatenate(axis=-1)([decoder_outputs, attention_context])
decoder_dense = Dense(len(target_char2idx), activation='softmax')(decoder_concat)

model = Model([encoder_inputs, decoder_inputs], decoder_dense)
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ------------------------------
# 5. Training
# ------------------------------

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_seq2seq_model.h5', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)

history = model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=128,
    epochs=50,
    validation_data=([val_encoder_input_data, val_decoder_input_data], val_decoder_target_data),
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])
decoder_state_input_h = Input(shape=(hidden_dim,))
decoder_state_input_c = Input(shape=(hidden_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

encoder_outputs_input = Input(shape=(None, hidden_dim))

decoder_inputs_single = Input(shape=(1,))
dec_emb_single = Embedding(len(target_char2idx), embedding_dim)(decoder_inputs_single)
decoder_outputs_single, state_h_single, state_c_single = decoder_lstm(
    dec_emb_single, initial_state=decoder_states_inputs)

attention_scores = Dot(axes=[2, 2])([decoder_outputs_single, encoder_outputs_input])
attention_weights = Activation('softmax')(attention_scores)
attention_context_single = Dot(axes=[2, 1])([attention_weights, encoder_outputs_input])
decoder_concat_single = Concatenate(axis=-1)([decoder_outputs_single, attention_context_single])
decoder_outputs_single = Dense(len(target_char2idx), activation='softmax')(decoder_concat_single)

decoder_model = Model(
    [decoder_inputs_single, encoder_outputs_input] + decoder_states_inputs,
    [decoder_outputs_single, state_h_single, state_c_single]
)

@tf.function
def encoder_inference(input_seq):
    return encoder_model(input_seq, training=False)

@tf.function
def decoder_inference(input_seq, enc_output, states):
    return decoder_model([input_seq, enc_output] + states, training=False)

def batch_decode_sequences(input_seqs, batch_size=32):
    """Batch decoding for faster inference"""
    num_samples = len(input_seqs)
    all_decoded = []
    
    for i in range(0, num_samples, batch_size):
        batch = input_seqs[i:i+batch_size]
        batch_decoded = []
        
        # Process batch
        enc_outputs, state_h, state_c = encoder_inference(batch)
        states = [state_h, state_c]
        
        # Initialize targets
        target_seqs = tf.fill([batch.shape[0], 1], target_char2idx['\t'])
        
        for _ in range(max_decoder_seq_length):
            output_tokens, h, c = decoder_inference(target_seqs, enc_outputs, states)
            sampled_token_indices = tf.argmax(output_tokens[:, -1, :], axis=-1).numpy()
            states = [h, c]
            target_seqs = tf.expand_dims(tf.convert_to_tensor(sampled_token_indices), 1)
            batch_decoded.append([target_idx2char[idx] for idx in sampled_token_indices])
            if all(idx == target_char2idx['\n'] or idx == target_char2idx['<PAD>'] 
                  for idx in sampled_token_indices):
                break
        for j in range(len(batch)):
            decoded = []
            for chars in zip(*batch_decoded):
                if chars[j] == '\n' or chars[j] == '<PAD>':
                    break
                decoded.append(chars[j])
            all_decoded.append(''.join(decoded))
    
    return all_decoded

# ------------------------------
# 6. Evaluation
# ------------------------------

def evaluate_accuracy(test_inputs, test_targets, batch_size=32):
    """Batch evaluation for faster processing"""
    decoded = batch_decode_sequences(test_inputs, batch_size)
    correct = sum(1 for d, t in zip(decoded, test_targets) if d == t.replace('\t', '').replace('\n', ''))
    return correct / len(test_inputs) * 100

# Sample predictions (fixed version)
print("\nSample Predictions:")
decoded_samples = batch_decode_sequences(test_encoder_input_data[:10])
for i in range(10):
    actual_text = test_target_texts[i].replace('\t', '').replace('\n', '')
    print(f"Source: {test_input_texts[i]}")
    print(f"Predicted: {decoded_samples[i]}")
    print(f"Actual: {actual_text}")
    print("---")