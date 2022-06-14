
import os
import numpy as np
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class Corpus():
    text = ""
    num_chars = 0
    vocab_size = 0
    dataset = None
    chars_from_ids = None
    ids_from_chars = None

    def __init__(self):
        poem_folder = "real_poems/"
        poets = ["Busch.txt", "Heine.txt", "Keller.txt"]
        for p in poets:
            filename = poem_folder + p
            file = open(filename, 'r')
            self.text = self.text + file.read() + "\n\n"
        self.num_chars = len(self.text)
        file.close()

    def process(self):
        text = self.text
        vocab = sorted(set(text))
        chars = tf.strings.unicode_split(text, input_encoding='UTF-8')

        self.ids_from_chars = tf.keras.layers.StringLookup(
            vocabulary=list(vocab), mask_token=None)
        ids = self.ids_from_chars(chars)
        self.vocab_size = len(self.ids_from_chars.get_vocabulary())

        self.chars_from_ids = tf.keras.layers.StringLookup(
            vocabulary=self.ids_from_chars.get_vocabulary(),
            invert=True,
            mask_token=None)
        chars = self.chars_from_ids(ids)

        all_ids = self.ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
        ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
        seq_length = 100
        sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

        def split_input_target(sequence):
            input_text = sequence[:-1]
            target_text = sequence[1:]
            return input_text, target_text  # input, label

        dataset = sequences.map(split_input_target)

        BATCH_SIZE = 64
        BUFFER_SIZE = 10000

        dataset = (
            dataset
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))

        self.dataset = dataset


class MyModel(tf.keras.Model):
    def __init__(self, corpus: Corpus, embedding_dim, rnn_units):
        super().__init__(self)
        self.corpus = corpus
        vocab_size = corpus.vocab_size
        self.dataset = corpus.dataset
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

    def train(self):
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.compile(optimizer='adam', loss=loss)
        checkpoint_dir = './training_checkpoints'
        checkpoint_path = os.path.join(checkpoint_dir, "ckpt_{epoch}")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                     filepath=checkpoint_path,
                                                     save_weights_only=True)

        EPOCHS = 500
        self.fit(self.dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


class PoemGenerator(tf.keras.Model):
    def __init__(self, model, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = model.corpus.chars_from_ids
        self.ids_from_chars = model.corpus.ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated
        skip_ids = self.ids_from_chars(['UNK'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(self.ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                              return_state=True)

        # Only use the last prediction
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state
        return predicted_chars, states

def get_poem_generator(train_new=False):
    if train_new:
        corpus = Corpus()
        corpus.process()

        model = MyModel(
            corpus=corpus,
            embedding_dim=256,
            rnn_units=1024,
        )
        model.train()
        poem_gen = PoemGenerator(model)
    else:
        poem_gen = tf.saved_model.load('poem_generator')

    return poem_gen


def generate_text(generator, start_string):
    result = start_string
    next_char = tf.constant([result])
    states = None
    num_sentences = 0
    while num_sentences < 3:
        next_char, states = generator.generate_one_step(next_char, states=states)
        decode_char = next_char[0].numpy().decode("utf-8")
        result += decode_char
        if decode_char == '.' or decode_char == '?' or decode_char == '!':
            num_sentences += 1
    return result


#poem_gen = get_poem_generator(train_new=True)
#result = generate_text(poem_gen, 'E')
#print(result)

#tf.saved_model.save(poem_gen, 'poem_generator')


def stylize(text):
    html_doc = """
<!DOCTYPE html>
<head>
 <title>Gedichte-KI</title>
 <style>
 body {
   background-image: url("background.jpg")
 }
 p {
   color: black;
   font-family: "Lucida Handwriting", cursive;
   font-size: 4vw;
   text-align: center
  }

    .first-letter {
      font-size: 12vw;
      line-height: 2vw;
    }
 </style>
</head>

<body>
  <div style="white-space: pre">
    <p>
<span class="first-letter">
"""

    html_doc += text[0]
    html_doc += "</span>"
    html_doc += text[1:]
    html_doc += """</p>
  </div>

</body>
<img src="feather.png" style="width:33vw; float:right; padding-right:20vw; margin-top:-7vw">
</html>"""

    return html_doc
