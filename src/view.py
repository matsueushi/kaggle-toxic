# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('submission.csv')


# %%
print(train[:100])

# %%
ret = pd.merge(test, submission)
print(ret[:100])


# %%
train[:100].to_csv('../input/train_first100.csv')
test[:100].to_csv('../input/test_first100.csv')

# %%
train_p = pd.read_csv('../input/train_prepro.csv')
test_p = pd.read_csv('../input/test_prepro.csv')
train_p[:100].to_csv('../input/train_prepro_first100.csv')
test_p[:100].to_csv('../input/test_prepro_first100.csv')
