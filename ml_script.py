# %%
# Importing modules and dependencies
import pandas as pd
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# %%
# Reading the data from CSV files
df = pd.read_excel(r'./Samlet_Data.xlsx')
pred = pd.read_excel(r'./predict.xlsx')

# %%
# Reading-in our data, creating a X and Y variable
x = df.drop(['ADHD', 'Person'], axis="columns")
y = df['ADHD']

# %%
# Using a train_test_split function that gives us an X train data frame
# and a Y train data frame, used to appropriately train our model
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# %%
# Building and compiling the model
model = Sequential(
    [Dense(units=1500, activation='sigmoid', input_dim=len(xtrain.columns)),
     Dense(units=500, activation='relu'),

     Dense(units=1, activation='sigmoid')
     ])

# Declaring our loss, optimizer and metrics in order for the model to know how far the estimations are to right evaluation,
# the optimizer is helping to reduce our loss and get us closer to our desired outcome and metrics are responsible to evaluate how the model is performing
model.compile(loss='binary_crossentropy',
              optimizer='Adamax', metrics='accuracy')

# %%
# Model fitting, declaring our features and our target, how long we want to train our model for
# and how large of a batch we want to pass before making an update
model.fit(xtrain, ytrain, epochs=1, batch_size=2,
          validation_data=(xtest, ytest))

# %%
# Predicting from predict.xlsx file
def test(x):
    if x > 0.5:
        return 1
    return 0

res = model.predict(pred)
res = pd.DataFrame(res)
res[0] = res[0].map(lambda x: test(x))
res

# %%
# Saving the results to results.xlsx
writer = pd.ExcelWriter('results.xlsx', engine='xlsxwriter')

res.to_excel(writer, sheet_name='Sheet1', index=False)
writer.save()
