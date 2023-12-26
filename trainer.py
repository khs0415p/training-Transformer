import os
import time
import tqdm
from transformer import Transformer
from optim import ScheduledAdam
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch
import numpy as np
from cosineannealing import CosineAnnealingWarmUpRestarts

class Trainer:
    def __init__(self, config, mode, tokenizer, name, train_loader=None, valid_loader=None):
        self.config = config
        self.device = self.config.device
        self.mode = mode
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.clip = self.config.clip

        # beam
        self.K = self.config.K

        self.model = Transformer(config).to(self.device)
        # self.optimizer = ScheduledAdam(optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=config.eps), self.config.d_embed, self.config.warm_steps)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-3, betas=(0.9, 0.999))
        self.scheduler = CosineAnnealingWarmUpRestarts(optimizer=self.optimizer, T_0=self.config.T_0, T_mult=self.config.T_mult, eta_max = self.config.eta_max, T_up = self.config.T_up, gamma= self.config.gamma)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.config.pad_id)

        if self.mode == "test":
            
            self.model.load_state_dict(torch.load(os.path.join(config.model_path, name + '_model.pt')))

        

    def train(self):
        print(self.model)
        print(f'The model has {self.model.count_params():,} trainable parameters')
        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in tqdm.tqdm(range(self.config.epochs), desc='Epoch'):
            self.model.train()

            epoch_loss = 0
            start_time = time.time()
            for batch in tqdm.tqdm(self.train_loader, desc='Train Batch'):
                src = batch[0].to(self.device)
                trg = batch[1].to(self.device)
                self.optimizer.zero_grad()

                _, logits = self.model(src, trg)

                loss = self.criterion(logits[:, :-1, :].reshape(-1, logits.size(-1)), trg[:, 1:].reshape(-1))
                loss.backward()
                epoch_loss += loss.item()
                clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()

            self.scheduler.step()
            train_loss = epoch_loss / len(self.train_loader)
            valid_loss = self.eval()

            end_time = time.time()

            print(f'Epoch {epoch+1} | Epoch Time {end_time - start_time:.2f}')
            print(f'Train loss {train_loss:.3f} | Valid_loss {valid_loss:.3f}')
            print('-'*20)
            
            
            if valid_loss < best_val_loss:
                print(f"Saving model ...")
                best_val_loss = valid_loss
                best_epoch = epoch + 1
            torch.save(self.model.state_dict(), os.path.join(self.config.model_path,'best_model.pt'))

        print(f'best loss : {best_val_loss:.3f}\t| best epoch : {best_epoch:3d}')

    def eval(self):
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for batch in tqdm.tqdm(self.valid_loader, desc='Valid Batch'):
                src = batch[0].to(self.config.device)
                trg = batch[1].to(self.config.device)

                _,logits = self.model(src, trg)
                

                loss = self.criterion(logits[:, :-1, :].reshape(-1, logits.size(-1)), trg[:, 1:].reshape(-1))
                epoch_loss += loss.item()

            for _ in range(5):
                idx = np.random.randint(0, trg.size(0))
                pred = self.tokenizer.decodeAsids(torch.argmax(logits[idx], dim=-1).tolist())
                source = self.tokenizer.decodeAsids(src[idx, :].tolist())
                pred_token = torch.argmax(logits[idx], dim=-1).tolist()
                target = self.tokenizer.decodeAsids(trg[idx, :].tolist())
            
                print(f'QUERY : {source}\nTRG : {target}\nPRED : {pred}\npred_token : {pred_token}')
                print()
                
            return epoch_loss / len(self.valid_loader)

    def inference(self, text):
        self.model.eval()

        q = self.tokenizer.encodeAsids(text)
        q = torch.LongTensor(q).unsqueeze(0).to(self.device)

        output = torch.LongTensor([self.config.bos_id]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            while True:            
                _,logits = self.model(q, output)
                pred = torch.argmax(logits[:, -1, :], dim=-1)
                
                if pred == self.config.eos_id:
                    break
                
                pred = pred.unsqueeze(0)
                output = torch.cat((output, pred), dim=1)
                if len(output[0]) >= 50:
                    break

            result = self.tokenizer.decodeAsids(output.squeeze(0).tolist())

        return result

    def beamsearch(self, logits, K):
        
        pred = F.softmax(logits, dim=-1)
        torch.max(pred, dim=-1,)


