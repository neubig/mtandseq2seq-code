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
        self.enc_fwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
        self.enc_bwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)

        self.dec_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE*2+EMBEDDINGS_SIZE, STATE_SIZE, model)

        self.input_lookup = model.add_lookup_parameters((INPUT_VOCAB_SIZE, EMBEDDINGS_SIZE))
        self.attention_w1 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*2))
        self.attention_w2 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*LSTM_NUM_OF_LAYERS*2))
        self.attention_v = model.add_parameters( (1, ATTENTION_SIZE))
        self.decoder_w = model.add_parameters( (OUTPUT_VOCAB_SIZE, STATE_SIZE))
        self.decoder_b = model.add_parameters( (OUTPUT_VOCAB_SIZE))
        self.output_lookup = model.add_lookup_parameters((OUTPUT_VOCAB_SIZE, EMBEDDINGS_SIZE))

    def save_to_disk(self, filename):
        dy.save(filename, [self.enc_fwd_lstm, self.enc_bwd_lstm, self.dec_lstm, self.input_lookup, self.attention_w1, self.attention_w2, self.attention_v, self.decoder_w, self.decoder_b, self.output_lookup])

    def load_from_disk(self, filename):
        (self.enc_fwd_lstm, self.enc_bwd_lstm, self.dec_lstm, self.input_lookup, self.attention_w1, self.attention_w2, self.attention_v, self.decoder_w, self.decoder_b, self.output_lookup) = dy.load(filename, self.model)


    def embed_sentence(self, sentence):
        #sentence = list(sentence)
        sentence = [input_vocab.w2i[c] for c in sentence]

        global input_lookup

        return [self.input_lookup[char] for char in sentence]


    def run_lstm(self, init_state, input_vecs):
        s = init_state

        out_vectors = []
        for vector in input_vecs:
            s = s.add_input(vector)
            out_vector = s.output()
            out_vectors.append(out_vector)
        return out_vectors


    def encode_sentence(self, sentence):
        sentence_rev = list(reversed(sentence))

        fwd_vectors = self.run_lstm(self.enc_fwd_lstm.initial_state(), sentence)
        bwd_vectors = self.run_lstm(self.enc_bwd_lstm.initial_state(), sentence_rev)
        bwd_vectors = list(reversed(bwd_vectors))
        vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

        return vectors


    def attend(self, input_mat, state, w1dt):
        w2 = dy.parameter(self.attention_w2)
        v = dy.parameter(self.attention_v)

        # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
        # w1dt: (attdim x seqlen)
        # w2dt: (attdim x attdim)
        w2dt = w2*dy.concatenate(list(state.s()))
        # att_weights: (seqlen,) row vector
        unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        att_weights = dy.softmax(unnormalized)
        # context: (encoder_state)
        context = input_mat * att_weights
        return context


    def decode(self, vectors, output):
        output = list(output)
        output = [output_vocab.w2i[c] for c in output]

        w = dy.parameter(self.decoder_w)
        b = dy.parameter(self.decoder_b)
        w1 = dy.parameter(self.attention_w1)
        input_mat = dy.concatenate_cols(vectors)
        w1dt = None

        last_output_embeddings = self.output_lookup[output_vocab.w2i[EOS]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE*2), last_output_embeddings]))
        loss = []

        for char in output:
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or w1 * input_mat
            vector = dy.concatenate([self.attend(input_mat, s, w1dt), last_output_embeddings])
            s = s.add_input(vector)
            out_vector = w * s.output() + b
            probs = dy.softmax(out_vector)
            last_output_embeddings = self.output_lookup[char]
            loss.append(-dy.log(dy.pick(probs, char)))
        loss = dy.esum(loss)
        return loss


    def generate(self, in_seq):
        embedded = self.embed_sentence(in_seq)
        encoded = self.encode_sentence(embedded)

        w = dy.parameter(self.decoder_w)
        b = dy.parameter(self.decoder_b)
        w1 = dy.parameter(self.attention_w1)
        input_mat = dy.concatenate_cols(encoded)
        w1dt = None

        last_output_embeddings = self.output_lookup[output_vocab.w2i[EOS]]
        s = self.dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_output_embeddings]))

        out = ''
        count_EOS = 0
        for i in range(len(in_seq)*2):
            if count_EOS == 2: break
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or w1 * input_mat
            vector = dy.concatenate([self.attend(input_mat, s, w1dt), last_output_embeddings])
            s = s.add_input(vector)
            out_vector = w * s.output() + b
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
        encoded = self.encode_sentence(embedded)
        return self.decode(encoded, output_sentence)

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
        for iteration in range(10):
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
    #parser.add_argument('--print_probs', action="store_true", help='whether to print the probabilities per word over the validation set')
    #parser.add_argument('--perform_train', action="store_true", help='whether to perform training')
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


