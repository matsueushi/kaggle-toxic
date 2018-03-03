# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np

# %%
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train[:100].to_csv('../input/train_first100.csv')
test[:100].to_csv('../input/test_first100.csv')

# %%
submission = pd.read_csv('submission.csv')
ret = pd.merge(test, submission)
print(ret[:100])


# %%
train_p = pd.read_csv('../input/train_prepro.csv')
test_p = pd.read_csv('../input/test_prepro.csv')
train_p[:100].to_csv('../input/train_prepro_first100.csv')
test_p[:100].to_csv('../input/test_prepro_first100.csv')


#%%
print(train_p[:100])
print(test_p[:100])


# %%
train_p.describe(include='all')


# %%
test_p.describe(include='all')


# %%
import matplotlib.pyplot as plt
LIST_CLASSES = ['toxic', 'severe_toxic',
                'obscene', 'threat', 'insult', 'identity_hate']
ind = np.arange(len(LIST_CLASSES))
fix, ax = plt.subplots()
sum_labels = [train_p[l].sum() for l in LIST_CLASSES]
plot_bars = plt.bar(ind, sum_labels)
ax.set_title('Toxicity types')
ax.set_xticks(ind)
ax.set_xticklabels(LIST_CLASSES)


# %%
