#Tuning de drop rate sur 20 epochs et 2 blocks, avec le test fabriqué à partir de 10% du train. Résultat : 0.20.
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
nombre_de_blocks = [2,3,4,2,3,4]
recall = [0.843, 0.830, 0.836, 0.903, 0.906, 0.948]
training_set = ['train', 'train', 'train', 'train_augmented', 'train_augmented', 'train_augmented']
ar = np.transpose([nombre_de_blocks, recall, training_set])
df = pd.DataFrame(ar, columns = ['nombre_de_blocks', 'score_f1', 'training_set'])
df['nombre_de_blocks'] = df['nombre_de_blocks'].astype(float)
df['score_f1'] = df['score_f1'].astype(float)
sns.lineplot(x="nombre_de_blocks", y="score_f1", hue='training_set', data=df)
plt.show()
