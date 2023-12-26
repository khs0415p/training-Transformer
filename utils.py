import pandas as pd
from sklearn.model_selection import train_test_split
import os

def data_split(path:str):
    df = pd.read_csv(path)
    df.drop('label', axis=1, inplace=True)

    train, valid = train_test_split(df, test_size=0.2, random_state=1234)
    train.to_csv('./data/train', index=False, header=False)
    valid.to_csv('./data/valid', index=False, header=False)

    if not os.path.exists('./vocab'):
        os.makedirs('./vocab', exist_ok=True)

    # train vocab
    out = open('./vocab/vocab_data', 'w')
    with open('./data/train', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace(',', '\n', 1)
            out.write(line.lstrip())


if __name__=="__main__":
    data_split('./data/ChatbotData.csv')
    