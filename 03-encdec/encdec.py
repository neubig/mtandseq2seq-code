import dynet as dy
import random
import argparse
import util

EOS = "<EOS>"


INPUT_VOCAB_SIZE = 0
OUTPUT_VOCAB_SIZE = 0
LSTM_NUM_OF_LAYERS = 1
EMBEDDINGS_SIZE = 32
STATE_SIZE = 32
ATTENTION_SIZE = 32


class EncDecModel:
    def __init__(self, model, LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, INPUT_VOCAB_SIZE, OUTPUT_VOCAB_SIZE):
        self.model = model
        self.lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
        self.input_lookup = model.add_lookup_parameters((INPUT_VOCAB_SIZE, EMBEDDINGS_SIZE))
        self.decoder_w = model.add_parameters( (OUTPUT_VOCAB_SIZE, STATE_SIZE))
        self.decoder_b = model.add_parameters( (OUTPUT_VOCAB_SIZE))
        self.output_lookup = model.add_lookup_parameters((OUTPUT_VOCAB_SIZE, EMBEDDINGS_SIZE))

    def save_to_disk(self, filename):
        dy.save(filename, [self.lstm, self.input_lookup, self.decoder_w, self.decoder_b, self.output_lookup])

    def load_from_disk(self, filename):
        (self.lstm, self.input_lookup, self.decoder_w, self.decoder_b, self.output_lookup) = dy.load(filename, self.model)


    def embed_sentence(self, sentence):
        sentence = [input_vocab.w2i[c] for c in sentence]
        return [self.input_lookup[char] for char in sentence]


    def run_lstm(self, init_state, input_vecs):
        s = init_state
        for vector in input_vecs:
            s = s.add_input(vector)
        return s


    def encode_sentence(self, sentence):
        sentence_rev = list(reversed(sentence))
        state = self.run_lstm(self.lstm.initial_state(), sentence)
        return state


    def decode(self, output, state):
        output = list(output)
        output = [output_vocab.w2i[c] for c in output]

        w = dy.parameter(self.decoder_w)
        b = dy.parameter(self.decoder_b)

        # Feed in EOS as input to denote the start decoding
        last_output_embedding = self.output_lookup[output_vocab.w2i[EOS]]
        loss = []

        for char in output:
            input_vector = last_output_embedding
            state = state.add_input(input_vector)
            out_vector = w * state.output() + b
            probs = dy.softmax(out_vector)
            last_output_embeddings = self.output_lookup[char]
            loss.append(-dy.log(dy.pick(probs, char)))
        loss = dy.esum(loss)
        return loss


    def generate(self, in_seq):
        embedded = self.embed_sentence(in_seq)
        state = self.encode_sentence(embedded)

        w = dy.parameter(self.decoder_w)
        b = dy.parameter(self.decoder_b)

        last_output_embedding = self.output_lookup[output_vocab.w2i[EOS]]

        out = ''
        count_EOS = 0
        for i in range(len(in_seq)*2):
            if count_EOS == 2: break
            inp_vector = last_output_embedding
            state = state.add_input(inp_vector)
            out_vector = w * state.output() + b
            probs = dy.softmax(out_vector).vec_value()
            next_char = probs.index(max(probs))
            last_output_embeddings = self.output_lookup[next_char]
            if output_vocab.i2w[next_char] == EOS:
                count_EOS += 1
                continue

            out += output_vocab.i2w[next_char]
        return out


    def get_loss(self, input_sentence, output_sentence):
        dy.renew_cg()
        embedded = self.embed_sentence(input_sentence)
        s = self.encode_sentence(embedded)
        return self.decode(output_sentence, s)

    def eval(self, inputs, outputs):
        N = len(inputs)
        correct = 0.0
        for i in range(N):
            prediction = self.generate(inputs[i])
            if i < 5:
                print(f"\t{' '.join(inputs[i])}\t{prediction}\t{''.join(outputs[i][:-1])}")
            if prediction == ''.join(outputs[i][:-1]):
                correct += 1
        accuracy = correct/N
        return accuracy


    def train(self, train, dev):
        train_i = train[0]
        train_o = train[1]
        dev_i = dev[0]
        dev_o = dev[1]

        prev_dev_acc = 0
        trainer = dy.SimpleSGDTrainer(self.model)
        N = len(train_i)
        ids = list(range(N))
        for iteration in range(50):
            random.shuffle(ids)
            for i in ids:
                loss = self.get_loss(train_i[i], train_o[i])
                loss_value = loss.value()
                loss.backward()
                trainer.update()
            print(f"Total Loss at Iteration {iteration} : {loss_value}")
            # Eval on dev
            dev_acc = self.eval(dev_i, dev_o)
            print(f"Dev accuracy at iteration {iteration} : {dev_acc}")
            if dev_acc > prev_dev_acc:
                prev_dev_acc = dev_acc
                self.save_to_disk("models/inflection.model")



            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='Path to the corpus file.')
    parser.add_argument('dev', help='Path to the validation corpus file.')
    parser.add_argument('test', help='Path to the test corpus file.')
    args, unknown = parser.parse_known_args()

    train_corpus = util.read_inflection_data(args.train, end=EOS)
    dev_corpus = util.read_inflection_data(args.dev, end=EOS)
    test_corpus = util.read_inflection_data(args.test, end=EOS)

    input_vocab = util.Vocab.from_corpus(train_corpus[0])
    output_vocab = util.Vocab.from_corpus(train_corpus[1])

    INPUT_VOCAB_SIZE = input_vocab.size()
    print(f"INPUT VOCAB SIZE: {INPUT_VOCAB_SIZE}")
    OUTPUT_VOCAB_SIZE = output_vocab.size()
    print(f"OUTPUT VOCAB SIZE: {OUTPUT_VOCAB_SIZE}")

    model = dy.Model()
    inflectionModel = EncDecModel(model, LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, INPUT_VOCAB_SIZE, OUTPUT_VOCAB_SIZE)

    inflectionModel.train(train_corpus, dev_corpus)

    # Load the best model
    inflectionModel.load_from_disk("models/inflection.model")

    # Test on test data
    test_acc = inflectionModel.eval(test_corpus[0], test_corpus[1])
    print(f"Accuracy on test: {test_acc}")


