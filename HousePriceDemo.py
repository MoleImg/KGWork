# coding=UTF-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_train = pd.read_csv("HousePrice/train.csv")
data_test = pd.read_csv("HousePrice/test.csv")

# Visualization
# plt.subplot2grid((3,3), (0,0))
# xticks = range(0, 800000, 100000)
# data_train.SalePrice.plot(kind='bar')
# plt.xlabel("ID")
# plt.ylabel("Price")

data_train.MoSold.value_counts().plot(kind='bar')
plt.xlabel("Month")
plt.ylabel("Frequency")

plt.show()