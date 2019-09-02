import dynet as dy
import time
import random

LAYERS = 2
INPUT_DIM = 256 #50  #256
HIDDEN_DIM = 256 # 50  #1024
VOCAB_SIZE = 0

from collections import defaultdict
from itertools import count
import argparse
import sys
import util
import math

class RNNLanguageModel:
    def __init__(self, model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=dy.SimpleRNNBuilder):
        self.builder = builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)

        self.lookup = model.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))
        self.R = model.add_parameters((VOCAB_SIZE, HIDDEN_DIM))
        self.bias = model.add_parameters((VOCAB_SIZE))

    def save_to_disk(self, filename):
        dy.save(filename, [self.builder, self.lookup, self.R, self.bias])

    def load_from_disk(self, filename):
        (self.builder, self.lookup, self.R, self.bias) = dy.load(filename, model)
        
    def build_lm_graph(self, sent):
        dy.renew_cg()
        init_state = self.builder.initial_state()

        R = dy.parameter(self.R)
        bias = dy.parameter(self.bias)
        errs = [] # will hold expressions
        es=[]
        state = init_state
        for (cw,nw) in zip(sent,sent[1:]):
            # assume word is already a word-id
            x_t = dy.lookup(self.lookup, int(cw))
            state = state.add_input(x_t)
            y_t = state.output()
            r_t = bias + (R * y_t)
            err = dy.pickneglogsoftmax(r_t, int(nw))
            errs.append(err)
        nerr = dy.esum(errs)
        return nerr
    
    def predict_next_word(self, sentence):
        dy.renew_cg()
        init_state = self.builder.initial_state()
        R = dy.parameter(self.R)
        bias = dy.parameter(self.bias)
        state = init_state
        for cw in sentence:
            # assume word is already a word-id
            x_t = dy.lookup(self.lookup, int(cw))
            state = state.add_input(x_t)
        y_t = state.output()
        r_t = bias + (R * y_t)
        prob = dy.softmax(r_t)
        return prob
    
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

    def get_ppl(self, sents):
        loss = 0.0
        chars = 0.0
        for sent in sents:
            isent = [vocab.w2i[w] for w in sent]
            errs = lm.build_lm_graph(isent)
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

    train = util.CharsCorpusReader(args.train, begin="<s>")
    dev = util.CharsCorpusReader(args.dev, begin="<s>")
    test = util.CharsCorpusReader(args.test, begin="<s>")

    vocab = util.Vocab.from_corpus(train)
    
    VOCAB_SIZE = vocab.size()

    model = dy.Model()
    trainer = dy.SimpleSGDTrainer(model, learning_rate=1.0)

    lm = RNNLanguageModel(model, LAYERS, INPUT_DIM, HIDDEN_DIM, VOCAB_SIZE, builder=dy.LSTMBuilder)

    train = list(train)

    prev_dev_ppl = 100000

    
    chars = loss = 0.0
    for ITER in range(100):
        random.shuffle(train)
        for i,sent in enumerate(train):
            _start = time.time()
            if i % 200 == 0:
                trainer.status()
                if chars > 0: print(loss / chars,)
                for _ in range(1):
                    samp = lm.sample(first=vocab.w2i["<s>"],stop=vocab.w2i["\n"])
                    print("".join([vocab.i2w[c] for c in samp]).strip())
                
                devppl = lm.get_ppl(dev)
                print(f"DEV ppl: {devppl}")
                if devppl < prev_dev_ppl:
                    lm.save_to_disk("LSTMLanguageModel.model")
                    prev_dev_ppl = devppl
                loss = 0.0
                chars = 0.0
                
            chars += len(sent)-1
            isent = [vocab.w2i[w] for w in sent]
            errs = lm.build_lm_graph(isent)
            loss += errs.scalar_value()
            errs.backward()
            trainer.update()
            #print "TM:",(time.time() - _start)/len(sent)
        print("ITER {}, loss={}".format(ITER, loss))
        trainer.status()
    

    #lm.save_to_disk("LSTMLanguageModel.model")

    print("loading the saved model...")
    lm.load_from_disk("LSTMLanguageModel.model")
    samp = lm.sample(first=vocab.w2i["<s>"],stop=vocab.w2i["\n"])
    print("".join([vocab.i2w[c] for c in samp]).strip())

    test_ppl = lm.get_ppl(test)
    print(f"Test perplexity: {test_ppl}")

 
