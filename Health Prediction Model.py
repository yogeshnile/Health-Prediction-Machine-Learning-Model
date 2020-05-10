# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
"""
### Import DataSet
"""

# %%
info = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')

info.head()

# %%
"""
### Convert Gender in to number format
"""

# %%
info['Gender'] = np.where((info.Gender == 'Male'), '0' , info['Gender'])

info['Gender'] = np.where((info.Gender == 'Female'), '1' , info['Gender'])

# %%
# in Gender Column Male = 0 and Female = 1

# %%
info.dtypes

# %%
info['Gender'] = info['Gender'].astype(int)

# %%
info.dtypes

# %%
info.describe()

# %%
"""
### Check Missing Value
"""

# %%
info.shape

# %%
info.info()

# %%
info['Gender'].value_counts()

# %%
info['Index'].value_counts()

# %%
"""
### View Regression in Fig.
"""

# %%
info.hist(bins = 50, figsize=(20,15))
plt.show()

# %%
"""
### Divide Train and Test Dataset
"""

# %%
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(info, test_size=0.2, random_state=42)

print(len(train_set))
print(len(test_set))

# %%
info = train_set.copy()

# %%
"""
### Correlations
"""

# %%
cor_data = info.corr()

cor_data['Index'].sort_values(ascending=False)

# %%
from pandas.plotting import scatter_matrix

scatter_matrix(info, figsize=(12,8))

# %%
"""
### Change a Feature and Label
"""

# %%
info = train_set.drop('Index', axis=1)
info_lable = train_set['Index'].copy()

# %%
"""
### SK-Learn Pipeline
"""

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
])

# %%
info_num_tr = my_pipeline.fit_transform(info)

# %%
info_num_tr.shape

# %%
"""
### Select LinearRegression Model
"""

# %%
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(info_num_tr, info_lable)

# %%
"""
#### Check Model
"""

# %%
some_data = info.iloc[:10]
some_label = info_lable.iloc[:10]

#for Checking purpose

# %%
prepared_data = my_pipeline.transform(some_data)

# %%
model.predict(prepared_data)

# %%
list(some_label)

# %%
"""
#### Evaluating the Model
"""

# %%
from sklearn.metrics import mean_squared_error

info_prediction = model.predict(info_num_tr)
mse = mean_squared_error(info_lable, info_prediction)
rmse = np.sqrt(mse)

# %%
rmse

# %%
"""
### Select DecisionTreeRegressor Model
"""

# %%
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
model.fit(info_num_tr, info_lable)

# %%
prepared_data = my_pipeline.transform(some_data)

model.predict(prepared_data)

# %%
list(some_label)

# %%
from sklearn.metrics import mean_squared_error

info_prediction = model.predict(info_num_tr)
mse = mean_squared_error(info_lable, info_prediction)
rmse = np.sqrt(mse)
rmse

# %%
## Overfitting

# %%
"""
##### Cross Validation
"""

# %%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, info_num_tr, info_lable, scoring='neg_mean_squared_error', cv=10)
rmse_scores = np.sqrt(-scores)
rmse_scores

# %%
"""
### Select SVM Model
"""

# %%
from sklearn import svm

model = svm.SVR()
model.fit(info_num_tr, info_lable)

# %%
prepared_data = my_pipeline.transform(some_data)

model.predict(prepared_data)

# %%
list(some_label)

# %%
from sklearn.metrics import mean_squared_error

info_prediction = model.predict(info_num_tr)
mse = mean_squared_error(info_lable, info_prediction)
rmse = np.sqrt(mse)
rmse

# %%
"""
### Select Bayesian Ridge Regression Model
"""

# %%
from sklearn import linear_model

model = linear_model.BayesianRidge()
model.fit(info_num_tr, info_lable)


# %%
prepared_data = my_pipeline.transform(some_data)

model.predict(prepared_data)

# %%
list(some_label)

# %%
from sklearn.metrics import mean_squared_error

info_prediction = model.predict(info_num_tr)
mse = mean_squared_error(info_lable, info_prediction)
rmse = np.sqrt(mse)
rmse

# %%
"""
### Select SGDRegressor Model
"""

# %%
from sklearn import linear_model

model = linear_model.SGDRegressor()
model.fit(info_num_tr, info_lable)

# %%
prepared_data = my_pipeline.transform(some_data)

model.predict(prepared_data)

# %%
list(some_label)

# %%
from sklearn.metrics import mean_squared_error

info_prediction = model.predict(info_num_tr)
mse = mean_squared_error(info_lable, info_prediction)
rmse = np.sqrt(mse)
rmse

# %%
"""
### Select Ridge regression Model
"""

# %%
from sklearn import linear_model

model = linear_model.Ridge()
model.fit(info_num_tr, info_lable)

# %%
prepared_data = my_pipeline.transform(some_data)

model.predict(prepared_data)

# %%
list(some_label)

# %%
from sklearn.metrics import mean_squared_error

info_prediction = model.predict(info_num_tr)
mse = mean_squared_error(info_lable, info_prediction)
rmse = np.sqrt(mse)
rmse

# %%
"""
### Select RandomForestRegressor Model
"""

# %%
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(info_num_tr, info_lable)

# %%
prepared_data = my_pipeline.transform(some_data)

model.predict(prepared_data)

# %%
list(some_label)

# %%
from sklearn.metrics import mean_squared_error

info_prediction = model.predict(info_num_tr)
mse = mean_squared_error(info_lable, info_prediction)
rmse = np.sqrt(mse)
rmse

# %%
"""
## Export Model
"""

# %%
from joblib import dump, load

dump(model, 'Health Prediction.joblib')

# %%
"""
## Testing Model
"""

# %%
x_test = test_set.drop('Index', axis=1)
y_test = test_set['Index'].copy()

x_test_prepared = my_pipeline.transform(x_test)
final_predictions = model.predict(x_test_prepared)

# %%
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

# %%
final_rmse

# %%
"""
## Exprot data for testing

"""

# %%
test_set.to_csv('test data.csv', index=False)

# %%
