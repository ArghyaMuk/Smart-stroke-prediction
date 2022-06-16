# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import dataseet
ds=pd.read_csv("/content/healthcare-dataset-stroke-data.csv")
# after read the datasheet 
ds
"""# Exploratory Data Analysis"""
# Checking the data like (How many row and columns we have in this current dataSheet)
ds.shape
# checking data types in This Current DataSheet
ds.info()
# count the null value in this current datasheet
ds.isna().sum()
# Check the 
ds.describe()
# Copy the dataset into new variable
ds1=ds.copy()
# id features does not play imp role, show we drop the id column
ds1.drop(['id'],axis=1, inplace=True)
"""# **Explore the Categorical Features**"""
categorical_features=[feature for feature in ds1.columns if ((ds1[feature].dtypes=='O'))]
categorical_features
# Check the highest mumber of catagorical values
for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(ds1[feature].unique()))
    )
#check count based on categorical features
plt.figure(figsize=(20,80), facecolor='white')
plotnumber =1
for categorical_feature in categorical_features:
    ax = plt.subplot(10,2,plotnumber)
    sns.countplot(x=categorical_feature,data=ds1)
    plt.xlabel(categorical_feature)
    plt.title('categorical_feature')
    plotnumber+=1
plt.show()
sns.catplot(x='stroke',hue='gender', palette='GnBu',kind='count',data=ds1)
plt.title('Gender vs Stroke')
sns.countplot(x='stroke',hue='ever_married', palette='GnBu',data=ds1)
plt.title('Marride vs Stroke')
sns.countplot(x='stroke',hue='work_type', palette='GnBu',data=ds1)
 plt.title('Work Type vs Stroke')
sns.countplot(x='stroke',hue='smoking_status', palette='GnBu',data=ds1)
 plt.title('Smoking Status vs Stroke')
# Boxplot:
plt.figure(figsize=(15,12))
ds1.plot(kind='box', subplots=True, layout=(2,3), figsize=(20, 10))
plt.show()
numerical_features=[feature for feature in ds1.columns if ((ds1[feature].dtypes!='O'))]
numerical_features
# Categorized discrete features
discrete_feature=[feature for feature in numerical_features if len(ds1[feature].unique())<3]
print("Discrete Variables Count: {}".format(len(discrete_feature)))
discrete_feature
sns.barplot(x='heart_disease',y='stroke',data=ds1)
plt.title('Heart Diseases vs Stroke')
sns.barplot(x='hypertension',y='stroke',data=ds1)
plt.title('Hypertension vs Stroke')
# Categorized cotinuous features
continuous_features=[feature for feature in numerical_features if feature not in discrete_feature+['stroke']]
print("Continuous feature Count {}".format(len(continuous_features)))
continuous_features
# Checking continuous_features is symmetric/skew-symmetric using visualization
plt.figure(figsize=(20,60), facecolor='white')
plotnumber =1
for continuous_feature in continuous_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.distplot(ds1[continuous_feature])
    plt.xlabel(continuous_feature)
    plotnumber+=1
plt.show()
ds2=ds1.copy()
# Replace Missing Values with Median
ds2['bmi'] = ds2['bmi'].fillna(ds2['bmi'].median(),inplace=False)
"""# **Encoding**"""
# Using LabelEncoding
from sklearn.preprocessing import LabelEncoder
en=LabelEncoder()
work_type = en.fit_transform(ds2['work_type'])
smoking_status = en.fit_transform(ds2['smoking_status'])
gender = en.fit_transform(ds2['gender'])
ever_married = en.fit_transform(ds2['ever_married'])
Residence_type = en.fit_transform(ds2['Residence_type'])
ds2['work_type']=work_type
ds2['smoking_status']=smoking_status
ds2['gender'] = gender
ds2['ever_married'] = ever_married
ds2['Residence_type']= Residence_type
ds2.head()
ds2.info()
"""# Imbalance Data Handling"""
ds3=ds2.copy()
ds3.stroke.value_counts()
X = ds3.drop('stroke', axis=1)
y = ds3['stroke']
y.value_counts()
sns.countplot(x=y)
# # Import RandomOverSampler Function
from imblearn.over_sampling import RandomOverSampler
oversampler = RandomOverSampler()
X_oversampler, y_oversampler = oversampler.fit_resample(X, y)
# import collection libary
# Before Appling SMOTE function
from collections import Counter
print('Before SMOTE: ',Counter(y))
# After Appling RandomOverSampler Function
print('After SMOTE: ',Counter(y_oversampler))
# Spliting Data for Traing & Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_oversampler, y_oversampler, test_size=0.2, random_state=0)
print(f'Train : {X_train.shape}')
print(f'Test: {X_test.shape}')
# Heatmap
plt.figure(figsize=(17, 15))
corr_mask = np.triu(ds3.corr())
h_map = sns.heatmap(ds3.corr(), mask=corr_mask)
h_map
"""# Normalize"""
# Standardization Using StandarScaler
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train_std=std.fit_transform(X_train)
X_test_std=std.transform(X_test)
X_train_std
import pickle
pickle.dump(std, open('StandarScaler.pkl', 'wb'))
"""# Model Creation"""
# Import Libary for Classification Reports
from sklearn.metrics import classification_report, confusion_matrix
# Function for Confusion martix  for every models
def col_max(model):
  df_cm = pd.DataFrame(model, index = ['Stroke', 'Normal'],
                                columns = ['Stroke', 'Normal'])

  p = sns.heatmap(df_cm, annot=True, cmap="YlGnBu" ,fmt='g')
  plt.title('Confusion matrix', y=1.1)
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
# ===========================================
# for model results and model name
models_results = np.zeros(5)
m_name=['Decision Tree','Random Forest','Gradient Boosting','KNeighbors Classifier','Artificial Neural networks']
"""# DecisionTreeClassifier"""
from sklearn.tree import DecisionTreeClassifier
dtc= DecisionTreeClassifier()
DTC=dtc.fit(X_train_std, y_train).predict(X_test_std)
print('Train Accuracy : {:.5f}'.format(dtc.score(X_train_std, y_train)))
print('Test Accuracy : {:.5f}'.format(dtc.score(X_test_std, y_test)))
print('=============================================================')
print(classification_report(y_test, DTC))
print('=============================================================')
DTCm = confusion_matrix(y_test, DTC)
models_results[0]=dtc.score(X_test_std, y_test)
# confusion_matrix visualization for DecisionTreeClassifier
col_max(DTCm)
"""# RandomForestClassifier"""
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
RFC=rfc.fit(X_train_std, y_train).predict(X_test_std)
print('Train Accuracy : {:.5f}'.format(rfc.score(X_train_std, y_train)))
print('Test Accuracy : {:.5f}'.format(rfc.score(X_test_std, y_test)))
print('=============================================================')
print(classification_report(y_test, RFC))
print('=============================================================')
RFCm = confusion_matrix(y_test, RFC)
models_results[1]=rfc.score(X_test_std, y_test)
# confusion_matrix visualization for RandomForestClassifier
col_max(RFCm)
"""# GradientBoostingClassifier"""
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
GBC=gbc.fit(X_train_std, y_train).predict(X_test_std)
print('Train Accuracy : {:.5f}'.format(gbc.score(X_train_std, y_train)))
print('Test Accuracy : {:.5f}'.format(gbc.score(X_test_std, y_test)))
print('=============================================================')
print(classification_report(y_test, GBC))
print('=============================================================')
GBCm=(confusion_matrix(y_test, GBC))
models_results[2]=gbc.score(X_test_std, y_test)
# confusion_matrix visualization for GradientBoostingClassifier
col_max(GBCm)
"""# KNeighborsClassifier"""
from sklearn.neighbors import KNeighborsClassifier
knc= KNeighborsClassifier()
KNC=knc.fit(X_train_std, y_train).predict(X_test_std)
print('Train Accuracy : {:.5f}'.format(knc.score(X_train_std, y_train)))
print('Test Accuracy : {:.5f}'.format(knc.score(X_test_std, y_test)))
print('=============================================================')
print(classification_report(y_test, KNC))
print('=============================================================')
KNCm=(confusion_matrix(y_test, KNC))
models_results[3]=knc.score(X_test_std, y_test)
# confusion_matrix visualization for KNeighborsClassifier
col_max(KNCm)
"""# Artificial Neural networks (ANN)"""
# Libary for ANN
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
ANN = Sequential()
#ANN Model
ANN.add(Dense(64,activation='relu'))
ANN.add(Dense(32,activation='relu'))
ANN.add(Dense(32,activation='relu'))
ANN.add(Dense(32,activation='relu'))
ANN.add(Dense(8,activation='relu'))
ANN.add(Dense(4,activation='relu'))
ANN.add(Dense(4,activation='relu'))
ANN.add(Dense(2,activation='relu'))
ANN.add(Dense(1, activation='sigmoid'))
# # ANN Compiler
ANN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Early Stopping Callback 
callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=4 , min_delta=0.001)
# # ANN Model Fit
ANN.fit(x=X_train_std, y=y_train,
          validation_data=(X_test_std,y_test),
          batch_size=32,epochs=300,callbacks=[callback])
y_pred = ANN.predict(X_test_std)
y_pred = (y_pred > 0.5)
ANNm = confusion_matrix(y_test, y_pred)
models_results[4]=np.round(ANN.evaluate(X_test, y_test, verbose=0)[1], 3)
# confusion_matrix visualization for ArtificialNeuralNetworks
col_max(ANNm)
# we make a dataframe for models results 
df = pd.DataFrame(data=m_name,columns=['Model_Name'])
df1 = pd.DataFrame(data=models_results,columns=['Result'])
result = pd.concat([df,df1],axis=1)
result
g = sns.catplot(x='Model_Name', y='Result', data=result,
                height=6, aspect=3, kind='bar', legend=True)
g.fig.suptitle('Accuracy for each model', size=35, y=1.1)
ax = g.facet_axis(0,0)
ax.tick_params(axis='x', which='major', labelsize=20)
# for printing accuracy persentage
for p in ax.patches:
    ax.text(p.get_x() + 0.27,
            p.get_height() * 1.02,
           '{0:.2f}%'.format(p.get_height()*100),
            color='black',
            rotation='horizontal',
            size='x-large')
pred1 = rfc.predict(X_test_std)
pred1
y_test
diff = pd.DataFrame(np.c_[y_test,pred1],columns=['Actual','Predicted'])
diff
from sklearn.metrics import accuracy_score
dtc_acc = accuracy_score(pred1 , y_test)
dtc_acc
pickle.dump(rfc,open('model_RandomForest.pkl','wb'))
model = pickle.load(open('model_RandomForest.pkl','rb'))
model.predict(X_test_std)