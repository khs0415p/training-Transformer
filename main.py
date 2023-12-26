from config import Config
import os
import argparse
from functools import partial
from dataset import CustomDataset, collate_fn
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from trainer import Trainer


def main(args):
    
    config = Config(args.config_path)
    tokenizer = Tokenizer(config)
    config.load_vocab(tokenizer=tokenizer)
    
    if args.mode == 'train':
        
        print('vocab size : ', config.src_vocab_size - 4)
        train_data = CustomDataset(config.train_path, config, tokenizer)
        valid_data = CustomDataset(config.valid_path, config, tokenizer)
        train_loader = DataLoader(train_data, config.batch_size, shuffle=True, collate_fn=partial(collate_fn, config=config))
        valid_loader = DataLoader(valid_data, config.batch_size, shuffle=False, collate_fn=partial(collate_fn, config=config))

        trainer = Trainer(config, args.mode, tokenizer, args.name, train_loader=train_loader, valid_loader=valid_loader)
        trainer.train()

    else:
        trainer = Trainer(config, args.mode, tokenizer, args.name)
        print("Chat start!")
        while True:
            text = input()

            if text in ('종료', 'exit'):
                break
            else:
                result = trainer.inference(text)
                print(result + '\n')
                

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--name', '-n', type=str, default='best')
    parser.add_argument('--config_path', type=str, default='/home/hyunsookim/conversation/transformer/config.json',  help = "input config path e.g) ./config.json ")
    
    args = parser.parse_args()

    main(args)


