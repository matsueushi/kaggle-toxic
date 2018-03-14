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
print(std_dev.sort_values(by=['threat_stddev']))


# %%
modified_ave = test[['id']].copy()
threshold = 0.15
for c in LABELS:
    ambiguous_label = std_dev[c + '_stddev'] > threshold
    modified_ave[c] = 0
    modified_ave.loc[~ambiguous_label, c] = sub_ave.loc[~ambiguous_label, c]
    for out in outputs:
        modified_ave.loc[ambiguous_label, c] = np.maximum(
            modified_ave.loc[ambiguous_label, c], out.loc[ambiguous_label, c])
    print(c, ambiguous_label.sum())
modified_ave.to_csv('../modified_average.csv', index=False)

# %%
print(modified_ave.head())

# %%
modified_diff = modified_ave.copy()
modified_diff['preprocessed_comment_text'] = test['preprocessed_comment_text']
for c in LABELS:
    modified_diff[c] -= sub_ave[c]

# %%
for c in LABELS:
    print(modified_diff.sort_values(by=[c], ascending=False))
