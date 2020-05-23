#Tuning de drop rate sur 20 epochs et 2 blocks, avec le test fabriqué à partir de 10% du train. Résultat : 0.20.
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
drop_rate = [0.10, 0.15, 0.175, 0.20, 0.225, 0.25, 0.3, 0.35, 0.10, 0.15, 0.175, 0.20, 0.225, 0.25, 0.3, 0.35, 0.10, 0.15, 0.175, 0.20, 0.225, 0.25, 0.3, 0.35]
score = [0.9277, 0.9746, 0.9696, 0.9948, 0.9921, 0.9673, 0.9896, 0.9794, 1.0000, 0.9948, 0.9948, 0.9974, 0.9818, 1.0000, 0.9922, 0.9922, 0.96249416, 0.9845964,  0.98203836, 0.9960983,  0.98692313, 0.98337823, 0.99089829, 0.98575845]
metric = ['precision', 'precision','precision','precision','precision','precision','precision','precision', 'recall', 'recall','recall','recall','recall','recall','recall','recall','f1_score','f1_score','f1_score','f1_score','f1_score','f1_score','f1_score','f1_score']
ar = np.transpose(np.array([drop_rate, score, metric]))
df = pd.DataFrame(ar, columns = ['drop_rate', 'score', 'metric'])
df['drop_rate'] = df['drop_rate'].astype(float)
df['score'] = df['score'].astype(float)
sns.lineplot(x="drop_rate", y="score", hue='metric', style='metric', palette="ch:2.5,.25", data=df)
plt.show()
