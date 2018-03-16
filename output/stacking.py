# coding:utf-8
import os
import pandas as pd
import numpy as np

# %%
files = [x for x in os.listdir() if os.path.splitext(x)[-1] == '.csv']
print(len(files), 'files')
print(files)

# %%
test = pd.read_csv('../input/test_prepro.csv')
print(test.head())

# %%
outputs = [pd.read_csv(f) for f in files]


# %%
sub_ave = test[['id']].copy()
LABELS = ['toxic', 'severe_toxic',
          'obscene', 'threat', 'insult', 'identity_hate']
num = len(files)
for c in LABELS:
    sub_ave[c] = 0

for out in outputs:
    for c in LABELS:
        sub_ave[c] += out[c] / num

sub_ave.to_csv('../submission_average.csv', index=False)

# %%
std_dev = sub_ave.copy()
std_dev['preprocessed_comment_text'] = test['preprocessed_comment_text']
for c in LABELS:
    std_dev[c + '_stddev'] = 0

for out in outputs:
    for c in LABELS:
        std_dev[c + '_stddev'] += (out[c] - sub_ave[c])**2 / (num - 1)


# %%
print(std_dev.sort_values(by=['toxic_stddev']))
