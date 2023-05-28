import os
import mitdeeplearning as mdl
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from IPython import display as ipythondisplay

# Download the dataset
songs = mdl.lab1.load_training_data()
songs = [song.numpy().decode("utf-8") for song in songs]

# Finding all unique characters in the joined songs
songs_joined = '\n\n'.join(songs)
vocab = sorted(set(songs_joined))

# A mapping from character to unique index
char2idx = {u:i for i, u in enumerate(vocab)}

# A mapping from indices to characters
idx2char = np.array(vocab)

# Vectorizing the songs strings
def vectorize_string(string):
    vectorized_output = np.array([char2idx[char] for char in string])
    return vectorized_output

vectorized_songs = vectorize_string(songs_joined)

# Creating training examples and targets
def get_batch(vectorized_songs, seq_length, batch_size):
    # Length of the vectorized song string (number of rows in the array minus 1)
    n = vectorized_songs.shape[0] - 1

    # randomly choose starting indices for the examples in the training batch
    idx = np.random.choice(n-seq_length, batch_size)

    # A list of input sequences for the training batch
    input_batch = [vectorized_songs[i : i + seq_length] for i in idx]

    # A list of output sequences for the training batch
    output_batch = [vectorized_songs[i+1 : i + seq_length+1] for i in idx]

    # the true inputs and targets for the network training
    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
    return x_batch, y_batch

# Defining the model
def LSTM(rnn_units):
    return tf.keras.layers.LSTM(
        rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid',
        stateful=True,
    )

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        # Layer 1: Embedding layer to transform indices into dense vectors of a fixed embedding size
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        # Layer 2: LSTM with rnn_units number of units
        LSTM(rnn_units),
        # Layer 3: Dense (fully-connected) layer that transforms the LSTM output
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

# Instantiating the model
embedding_dim = 256
rnn_units = 1024
batch_size = 32
model = build_model(len(vocab), embedding_dim, rnn_units, batch_size)

# Defining the loss function
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# Compiling the model
model.compile(optimizer='adam', loss=loss)

# Defining the training parameters
num_training_iterations = 2000
batch_size = 4
seq_length = 100
learning_rate = 5e-3

# Creating a checkpoint directory
checkpoint_dir = './model/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_checkpoint")


# Function to generate a new song using the trained model
def generate_song():
    # Load the trained model
    model = load_model()
    
    # Predicting a generated song
    def generate_text(model, start_string, generation_length=1000):
        
        # COnverting the start string to bumbers
        input_eval = [char2idx[char] for char in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        
        # Empty string to store our results
        text_generated = []
        
        model.reset_states()
        tqdm._instances.clear()
        
        for i in tqdm(range(generation_length)):
            # Evaluating the input and generating the next character predictions
            predictions = model(input_eval)
            
            # Removing the batch dimensions
            predictions = tf.squeeze(predictions, 0)
            
            # Using multinominal distribution to sample
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
            
            # Passing the predictions along the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)
            
            # Adding the predicted character to the generated text
            text_generated.append(idx2char[predicted_id])
            
        return (start_string + ''.join(text_generated))

    # Generate a new song or music sample using the model    
    generated_text = generate_text(model, start_string='X', generation_length=1000)
    generated_songs = mdl.lab1.extract_song_snippet(generated_text)    
    for i, song in enumerate(generated_songs):
        # Synthesize the waveform from a song
        waveform = mdl.lab1.play_song(song)
        
        # If it is a valid song, let's play it
        if waveform:
            print(f"Generated song: {i}")
            ipythondisplay.display(waveform)    
    # Return the generated song
    return generated_songs


