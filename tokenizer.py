import tqdm
from typing import List
import os
import sentencepiece as spm
from config import Config
from konlpy.tag import Kkma
from collections import Counter

class Tokenizer:
    def __init__(self, config):
        self.config = config
        self.tokenizer_type = self.config.tokenizer_type
        
        if self.tokenizer_type in ['bpe', 'unigram', 'char']:
            self.prefix = config.vocab_file_path+self.config.tokenizer_type + '_' + str(self.config.src_vocab_size)
            self.sp = spm.SentencePieceProcessor()

            if os.path.isfile(self.prefix + '.model'):
                self.sp.load(self.prefix + '.model')

            else:
                self.train()

        else:
            self.kkma = Kkma()
            self.word2idx, self.idx2word = self.konlpy()


    def konlpy(self):
        
        word2idx = {'[PAD]':0, '[UNK]':1, '[BOS]':2, '[EOS]':3}
        idx2word = {0:'[PAD]', 1:'[UNK]', 2:'[BOS]', 3:'[EOS]'}
        idx = len(self.word2idx)

        count_dict = Counter()
        with open(self.config.train_path, 'r' , encoding='utf-8') as f:
            lines=f.readlines()
            for line in lines:
                line=line.replace('\t',' ')
                tokens = self.kkma.morphs(line, stem=True)
                tokens = [ token for token in tokens if token not in self.config.stopword]
                count_dict.update(tokens)

        for k, v in count_dict.items():
            if v >= self.config.min_freq:
                word2idx[k] = idx
                idx2word[idx] = k
                idx += 1

        return word2idx, idx2word


    def make_vocab_data(self):
        if not os.path.isdir(self.config.vocab_file_path):
            os.makedirs(self.config.vocab_file_path, exist_ok=True)
        
        out = open(self.config.vocab_file_path + 'vocab_data', 'w')
        with open(self.config.train_path, 'r') as f:
            lines = f.readlines()
            for line in tqdm.tqdm(lines, desc='make vocab data'):
                line = line.replace('\t', '\n')
                out.write(line.lstrip())

        out.close()

    def train(self):

        if not os.path.isfile(self.config.vocab_file_path + 'vocab_data'):
            self.make_vocab_data()
            
        spm.SentencePieceTrainer.train(
                f'--input={self.config.vocab_file_path + "vocab_data"} --model_prefix={self.prefix} --vocab_size={self.config.src_vocab_size+4}'
                f' --model_type={self.config.tokenizer_type}' +
                ' --character_coverage=1.0' +
                ' --max_sentence_length=999999' +
                ' --pad_id=0 --pad_piece=[PAD]' +
                ' --unk_id=1 --unk_piece=[UNK]' +
                ' --bos_id=2 --bos_piece=[BOS]' +
                ' --eos_id=3 --eos_piece=[EOS]'
            )

        self.sp.load(self.prefix + '.model')        


    def encodeAsids(self, text:str):
        if self.tokenizer_type == 'konlpy':
            text = self.kkma(text, stem = True)
            text_ids = [ self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['[UNK]'] for t in text ]
            return text_ids

        return self.sp.EncodeAsIds(text)

    def encodeAspiece(self, text:str):
        if self.tokenizer_type == 'konlpy':
            return self.kkma(text, stemp=True)
        return self.sp.EncodeAsPieces(text)

    def decodeAsids(self, idx:List[int]):
        if self.tokenizer_type == 'konlpy':
            token = [ self.idx2word[id] for id in idx]
            return ' '.join(token)
        return self.sp.DecodeIds(idx)

    def get_vocab_size(self):
        if self.tokenizer_type == 'konlpy':
            return len(self.word2idx)

        return self.sp.vocab_size()


if __name__ == "__main__":
    config = Config('./config.json')
    tk = Tokenizer(config)
    
