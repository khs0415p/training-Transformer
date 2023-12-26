import os
import torch
import sentencepiece as spm
import json

class Config:
    def __init__(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.__dict__.update(data)

        if os.path.exists('./vocab/'+data['tokenizer_type']+'_'+str(data['src_vocab_size'])+'.model'):
            self.load_vocab()

    def load_vocab(self):
        sp = spm.SentencePieceProcessor()
        sp.Load('./vocab/'+self.__dict__['tokenizer_type']+'_'+str(self.__dict__['src_vocab_size'])+'.model')

        params = {
                "vocab_size" : sp.vocab_size(),
                "bos_id" : sp.bos_id(),
                "eos_id" : sp.eos_id(),
                "pad_id" : sp.pad_id(),
                "unk_id" : sp.unk_id(),
                "device" : torch.device("cuda" if torch.cuda.is_available() else "cpu")
        }
        self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__

if __name__ == "__main__":
    config = Config('./config.json')
    print(config.vocab_size)
    
    
        