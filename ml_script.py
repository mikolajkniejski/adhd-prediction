# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import random
import tensorflow as tf

# %%

df = pd.read_excel(r'./Samlet_Data.xlsx')
pred = pd.read_excel(r'./predict.xlsx')

# %%
x = df.drop(['ADHD', 'Person'], axis="columns")
y = df['ADHD']

# %%
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# %%
model = Sequential(
    [Dense(units=1500, activation='sigmoid', input_dim=len(xtrain.columns)),
     Dense(units=500, activation='relu'),

     Dense(units=1, activation='sigmoid')
     ])

model.compile(loss='binary_crossentropy',
              optimizer='Adamax', metrics='accuracy')


# %%
model.fit(xtrain, ytrain, epochs=1, batch_size=2,
          validation_data=(xtest, ytest))

# %%
# Predict from predict.xlsx file


def test(x):
    if x > 0.5:
        return 1
    return 0


res = model.predict(pred)
res = pd.DataFrame(res)
res[0] = res[0].map(lambda x: test(x))
res


# %%
# Save to results.xlsx
writer = pd.ExcelWriter('results.xlsx', engine='xlsxwriter')


res.to_excel(writer, sheet_name='Sheet1', index=False)
writer.save()
