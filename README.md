# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
from scipy import stats
df=pd.read_csv("/content/bmi.csv")
df.head()
```
<img width="635" height="277" alt="image" src="https://github.com/user-attachments/assets/deb19fc2-beb9-46bd-9692-d9f87b49c4f3" />
```
df.dropna()
```
<img width="586" height="537" alt="image" src="https://github.com/user-attachments/assets/365854f4-c1ba-40ad-bdaf-82de9526118e" />
```
max_vals=np.max(np.abs(df[['Height','Height']]))
max_vals
```
<img width="616" height="122" alt="image" src="https://github.com/user-attachments/assets/a805f046-2874-4a72-87ab-97bb22185d86" />
```
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="593" height="467" alt="image" src="https://github.com/user-attachments/assets/96104bed-922f-4fc6-a8ed-8cef38f7be63" />
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="550" height="465" alt="image" src="https://github.com/user-attachments/assets/dcf9aa84-893f-45f9-9dc3-624dec76263c" />
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="543" height="472" alt="image" src="https://github.com/user-attachments/assets/16426714-1fae-4884-b5b6-f21b32765455" />
```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="783" height="472" alt="image" src="https://github.com/user-attachments/assets/d6080dfe-900b-4934-8dba-c536e04fee94" />
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
<img width="527" height="452" alt="image" src="https://github.com/user-attachments/assets/4271ef6e-335b-46ab-8d33-e3bcf76918d0" />
```
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset("tips")
tips.head()
```
<img width="611" height="243" alt="image" src="https://github.com/user-attachments/assets/67663c06-d65a-4ed8-909a-f15d5acb9667" />
```
contigency_table = pd.crosstab(tips["sex"],tips["time"])
print(contigency_table)
```
<img width="351" height="150" alt="image" src="https://github.com/user-attachments/assets/822f3a32-5e57-4101-b83b-24f05381c92a" />
```
chi2,p,_,_ = chi2_contingency(contigency_table)
print("Chi-square value:",chi2)
print("p-value:",p)
```
<img width="537" height="171" alt="image" src="https://github.com/user-attachments/assets/afa86a05-854b-4f64-b395-ca540fc5abb4" />
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

# Create a sample dataset
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': ['A', 'B', 'C', 'A', 'B'],
    'Feature3': [0, 1, 0, 1, 0],
    'Target': [0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# Separate features and target
X = df[['Feature1', 'Feature3']]
y = df['Target']

# SelectKBest with mutual_info_classif for feature selection
selector = SelectKBest(score_func=mutual_info_classif, k=1)
X_new = selector.fit_transform(X, y)

# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)

# Print the selected features
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
<img width="612" height="367" alt="image" src="https://github.com/user-attachments/assets/007118aa-b11b-464f-b508-d04d89374052" />


# RESULT:
Thus the python code to read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.
