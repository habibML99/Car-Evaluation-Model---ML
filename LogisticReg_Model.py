import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

eval_data = pd.read_csv('car.csv')
# Renaming Columns
eval_data.columns = ['buying', 'maintainence', 'doors', 'persons', 'luggage_spc', 'safety', 'overall_car']
# Converting Dataset to Datapoints
eval_data['buying'] = eval_data['buying'].map({'low': 1, 'med': 2, 'high': 3, 'vhigh': 4})
eval_data['maintainence'] = eval_data['maintainence'].map({'low': 1, 'med': 2, 'high': 3, 'vhigh': 4})
eval_data['doors'] = eval_data['doors'].map({'2': 2, '3': 3, '4': 4, '5more': 5})
eval_data['luggage_spc'] = eval_data['luggage_spc'].map({'small': 1, 'med': 2, 'big': 3})
eval_data['persons'] = eval_data['persons'].map({'2': 2, '4': 4, 'more': 5})
eval_data['safety'] = eval_data['safety'].map({'low': 1, 'med': 2, 'high': 3})
eval_data['overall_car'] = eval_data['overall_car'].map({'unacc': 1, 'acc': 2, 'good': 3, 'vgood': 4})

# Saving Updated DataFrame
eval_data.to_csv('Car Data.csv', index=False)
data = pd.read_csv('car eval.csv')

##### The data is highly Imbalanced/ Hence, using SMOTE to balance the datapoints #####

x = data.drop('overall_car', axis=1)
y = data['overall_car']

from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')

_X1, _Y1 = smote.fit_resample(x, y) # round 1
_X2_, _Y2_ = smote.fit_resample(_X1, _Y1) # round 2
X_data, Y_data = smote.fit_resample(_X2_, _Y2_) # round 3 (final because there were 3 minority classes) 

# Saving the Updated DataFrame
saving = pd.concat([X_data, Y_data], axis=1)
saving.to_csv('Balanced CarEval.csv')

# Ready the Data for training
Data = pd.read_csv('Balanced CarEval.csv')
X = Data.drop('overall_car', axis=1)
Y = Data['overall_car']

from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(
             X,Y,
             test_size=0.2,
             random_state=42, )

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(  
    penalty='elasticnet',  # Applying ElasticNet Regularization
    C=0.01,                # Strictness of Penalty - lower C more strictness
    l1_ratio=0.34,         # Ratio of Lasso-Ridge  or (lasso)l1 percentage 
    solver='saga',
    max_iter=1300 )

model.fit(X_train, y_train)
print(model.score(X_train, y_train))    # Model Performance in Training

y_pred = model.predict(X_test)
print('Accuracy: ',accuracy_score(y_test, y_pred),'\n') # Accuracy of the Model
print(f'The Classification Report of the model is : \n{classification_report(y_test,y_pred)}') # ALL report of model , inc. F1score, precision, recall, etc.

### Visualizing the results......
plt.figure(figsize=(10,8),facecolor='grey)
plt.scatter(X_test, y_test, label='Actual Results', color='green')
plt.plot(X_test, y_pred, label='Predicted Results', color='orange')
plt.gca().spines[['top','right']].set_visibility(False)
plt.show()



