import pickle
import matplotlib.pyplot as plt
import numpy as np
train_data = pickle.load(open("datas/train_tensors.p","rb"))

for signal, label in train_data:
  colr = 'b'
  if np.argmax(label) == 0: continue
  if np.argmax(label) == 1: colr = 'g'
  if np.argmax(label) == 2: colr = 'r'
  plt.scatter(signal[0], signal[1], s=80, c=colr)
plt.show()
