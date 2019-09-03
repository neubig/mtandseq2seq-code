import dynet as dy
import time
import random

LAYERS = 2
INPUT_DIM = 256 #50  #256
HIDDEN_DIM = 256 # 50  #1024
VOCAB_SIZE = 0
MB_SIZE = 50  # mini batch size

import argparse
from collections import defaultdict
from itertools import count
import sys
import util
import math

class RNNLanguageModel:
    def __init__(self, model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=dy.SimpleRNNBuilder):
        # Char-level LSTM (layers=2, input=256, hidden=128, model)
        self.builder = builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)
        # Lookup parameters for word embeddings
        self.lookup = model.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))
        # Softmax weights/biases on top of LSTM outputs
        self.R = model.add_parameters((VOCAB_SIZE, HIDDEN_DIM))
        self.bias = model.add_parameters((VOCAB_SIZE))

    def save_to_disk(self, filename):
        dy.save(filename, [self.builder, self.lookup, self.R, self.bias])

    def load_from_disk(self, filename):
        (self.builder, self.lookup, self.R, self.bias) = dy.load(filename, model)


    # Build the language model graph
    def BuildLMGraph(self, sents):
        dy.renew_cg()
        # initialize the RNN
        init_state = self.builder.initial_state()
        # parameters -> expressions
        R = dy.parameter(self.R)
        bias = dy.parameter(self.bias)

        S = vocab.w2i["</s>"]
        # get the cids and masks for each step
        tot_chars = 0
        cids = []
        masks = []

        for i in range(len(sents[0])):
            cids.append([(vocab.w2i[sent[i]] if len(sent) > i else S) for sent in sents])
            mask = [(1 if len(sent)>i else 0) for sent in sents]
            masks.append(mask)
            tot_chars += sum(mask)

        # start the rnn with "<s>"
        init_ids = cids[0]
        s = init_state.add_input(dy.lookup_batch(self.lookup, init_ids))

        losses = []

        # feed char vectors into the RNN and predict the next char
        for cid, mask in zip(cids[1:], masks[1:]):
            score = dy.affine_transform([bias, R, s.output()])
            loss = dy.pickneglogsoftmax_batch(score, cid)
            # mask the loss if at least one sentence is shorter
            if mask[-1] != 1:
                mask_expr = dy.inputVector(mask)
                mask_expr = dy.reshape(mask_expr, (1,), len(sents))
                loss = loss * mask_expr

            losses.append(loss)
            # update the state of the RNN
            cemb = dy.lookup_batch(self.lookup, cid)
            s = s.add_input(cemb)

        return dy.sum_batches(dy.esum(losses)), tot_chars


    def sample(self, first=1, nchars=0, stop=-1):
        res = [first]
        dy.renew_cg()
        state = self.builder.initial_state()

        R = dy.parameter(self.R)
        bias = dy.parameter(self.bias)
        cw = first
        while True:
            x_t = dy.lookup(self.lookup, cw)
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            ydist = dy.softmax(r_t)
            dist = ydist.vec_value()
            rnd = random.random()
            for i,p in enumerate(dist):
                rnd -= p
                if rnd <= 0: break
            res.append(i)
            cw = i
            if cw == stop: break
            if nchars and len(res) > nchars: break
        return res

    def print_probs(self, sent):
        dy.renew_cg()
        # initialize the RNN
        init_state = self.builder.initial_state()
        # parameters -> expressions
        R = dy.parameter(self.R)
        bias = dy.parameter(self.bias)

        # get the cids for each step
        tot_chars = 0
        cids = []
        for w in sent:
            cids.append(vocab.w2i[w])
        # start the rnn with "<s>"
        init_ids = cids[0]
        s = init_state.add_input(dy.lookup(self.lookup, init_ids))

        for cid in cids[1:]:
            score = dy.affine_transform([bias, R, s.output()])
            loss = dy.pickneglogsoftmax(score, cid)
            print(f"{vocab.i2w[cid]} {math.exp(-loss.value())}")
            cemb = dy.lookup(self.lookup, cid)
            s = s.add_input(cemb)


    def get_ppl(self, sents):
        loss = 0.0
        chars = 0.0
        for sent in sents:
            errs, _ = lm.BuildLMGraph([sent])
            loss += errs.scalar_value()
            chars += len(sent)-1
        return math.exp(loss/chars)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='Path to the corpus file.')
    parser.add_argument('dev', help='Path to the validation corpus file.')
    parser.add_argument('test', help='Path to the test corpus file.')
    parser.add_argument('--print_probs', action="store_true", help='whether to print the probabilities per word over the validation set')
    parser.add_argument('--perform_train', action="store_true", help='whether to perform training')
    args, unknown = parser.parse_known_args()

    train = util.CorpusReader(args.train, begin="<s>", end="</s>")
    dev = util.CorpusReader(args.dev, begin="<s>", end="</s>")
    test = util.CorpusReader(args.test, begin="<s>", end="</s>")

    vocab = util.Vocab.from_corpus(train)
    
    VOCAB_SIZE = vocab.size()
    print(f"VOCAB SIZE: {VOCAB_SIZE}")

    model = dy.Model()
    trainer = dy.SimpleSGDTrainer(model)

    #lm = RNNLanguageModel(model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=SimpleRNNBuilder)
    lm = RNNLanguageModel(model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=dy.LSTMBuilder)

    if args.perform_train:
        train = list(train)
        # Sort training sentences in descending order and count minibatches
        train.sort(key=lambda x: -len(x))
        train_order = [x*MB_SIZE for x in range(int((len(train)-1)/MB_SIZE + 1))]
        print(f"Created {len(train_order)} minibatches.")


        prev_dev_ppl = 100000
        # Perform training
        i = 0
        chars = loss = 0.0
        for ITER in range(100):
            random.shuffle(train_order)
            #_start = time.time()
            for sid in train_order: 
                i += 1
                if i % int(100) == 0:
                    trainer.status()
                    if chars > 0: print(loss / chars,)
                    for _ in range(1):
                        samp = lm.sample(first=vocab.w2i["<s>"],stop=vocab.w2i["</s>"])
                        print(" ".join([vocab.i2w[c] for c in samp]).strip())
                    
                    devppl = lm.get_ppl(dev)
                    print(f"DEV ppl: {devppl}")
                    if devppl < prev_dev_ppl:
                        lm.save_to_disk("LSTMLanguageModel-word-batch.model")
                        prev_dev_ppl = devppl
                    loss = 0.0
                    chars = 0.0

                # train on the minibatch
                errs, mb_chars = lm.BuildLMGraph(train[sid: sid + MB_SIZE])
                loss += errs.scalar_value()
                chars += mb_chars
                errs.backward()
                trainer.update()

            print("ITER",ITER,loss)
            trainer.status()

    print("loading the saved model...")
    lm.load_from_disk("LSTMLanguageModel-word-batch.model")
    samp = lm.sample(first=vocab.w2i["<s>"],stop=vocab.w2i["</s>"])
    print(" ".join([vocab.i2w[c] for c in samp]).strip())

    test_ppl = lm.get_ppl(test)
    print(f"Test perplexity: {test_ppl}")

    if args.print_probs:
        for sent in list(dev):
            lm.print_probs(sent)
            print()

