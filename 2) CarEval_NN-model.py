import pandas as pd
import numpy as np
from sklearn.matrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

data = pd.read_csv('data.csv')
