import os
from data_preprocessor import DataLoader

filename = os.path.join("../data", "sinewave.csv")
split = 0.8
cols = ["sinewave"]
seq_len = 50

data_loader = DataLoader(filename=filename, split=split, cols=cols)
x_train, y_train = data_loader.get_train_data(seq_len=seq_len, normalised=False)
print(x_train)


