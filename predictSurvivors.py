import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 1000)

# importing titanic train dataset in csv file
titanic_df = pd.read_csv("../PerfectGuide/1장/titanic/train.csv")
print(titanic_df.head().to_string())

# Columns meaning
titanic_df_column_dict = {
    'PassingerId': '탑승자 데이터 일련번호',
    'Survived': {
        '생존 여부': {
            0: "사망",
            1: "생존"
        }
    },
    'Pclass': {
        '티켓의 선실 등급': {
            1: '일등석',
            2: '이등석',
            3: '삼등석'
        }
    },
    'Name': '탑승자 이름',
    'Sex': '탑승자 성별',
    'Age': '탑승자 나이',
    'SibSp': '같이 탑승한 형제자매/배우자 인원수',
    'Parch': '같이 탑승한 부모님 또는 어린이 인원수',
    'Ticket': '티켓 번호',
    'Fare': '요금',
    'Cabin': '선실 번호',
    'Embarked': {
        '중간 정착 항구': {
            'C': 'Cherbourg',
            'Q': 'Queenstown',
            'S': 'Southampton'
        }
    }
}


# Define a function that returns distinctive value depends on the incoming age.
def get_category(age):
    cat = ''
    if age <= -1: cat = 'Unknown'
    elif age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else: cat = 'Elderly'

    return cat


# Define a function that processes multiple Label Encodings
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']
    le = LabelEncoder()
    for feature in features:
        le.fit(df[feature])
        df[feature] = le.transform(df[feature])

    return df


# Define a function that eliminates unnecessary features
def drop_features(df):
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return df


# Define a function that preprocesses Null values
def process_na(df):
    df['Age'].fillna(df['Age'].mean(), inplace=True)    # Substituting falsy values in Age column with average of the values in Age column
    df['Cabin'].fillna('N', inplace=True)                       # Substituting falsy values in Cabin column with 'N'
    df['Embarked'].fillna('N', inplace=True)                    # Substituting falsy values in Embarked column with 'N'
    return df


# Define a function that ensembles preprocesses
def transform_features(df):
    df = process_na(df)
    df = drop_features(df)
    df = format_features(df)
    return df


print('\n ### Train Data Information ### \n')
print(titanic_df.info(), '\n')
print(titanic_df.describe().to_string())

# Preprocessing Null columns
print('\nNumber of Null values in the Data Set before preprocessing :', titanic_df.isnull().sum().sum())
titanic_df = process_na(titanic_df)
print('Number of Null values in the Data Set after preprocessing :', titanic_df.isnull().sum().sum())

# EDA(Exploratory Data Analysis) of the important columns
# Extracting Column's with column type, object
print('Columns with data type, object :', titanic_df.dtypes[titanic_df.dtypes == 'object'].index.tolist())
print('\n### Distribution of the values of Sex column ###\n', titanic_df['Sex'].value_counts())
print('\n### Distribution of the values of Cabin column ###\n', titanic_df['Cabin'].value_counts())
print('\n### Distribution of the values of Embarked column ###\n', titanic_df['Embarked'].value_counts())

print('\n Values in Cabin column before transformation \n', titanic_df['Cabin'].head())
titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]
print('\n Values in Cabin column after transformation \n', titanic_df['Cabin'].head())

print('\n', titanic_df.groupby(['Sex', 'Survived'])['Survived'].count())
# Plot a bar plot about the number of survivors per gender
sns.barplot(x='Sex', y='Survived', data=titanic_df)
plt.show()
# Plot a bar plot about the number of survivors in each Pclass per gender
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df)   # hue is for color encoding
plt.show()
print("The last bar plot suggests the relationship between the gender and the number of survivors.\n",
      "It suggests that females were more likely to survive than the males did.")

# Set the bar plots' bars' figure to larger size
plt.figure(figsize=(10,6))

# Set the X values to order in ascending order of age
group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Elderly']

# Assign categorical values returned by get_category method to a new column, Age_cat
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x))

sns.barplot(x='Age_cat', y='Survived', hue='Sex', data=titanic_df, order=group_names)
plt.show()
print('The last bar plot suggests that the relationship between the age and the probability of survive \n',
      'Younger the passenger was, more likely to survive')
titanic_df.drop('Age_cat', axis=1, inplace=True)

titanic_df = format_features(titanic_df)
print("\n### Label Encoded version of titanic train Dataset ###\n")
print(titanic_df.head().to_string())

titanic_df = drop_features(titanic_df)
print("\n### Titanic train Dataset without unnecessary features ###\n")
print(titanic_df.head().to_string())

### ************************************************************************************************************************

new_titanic_df = pd.read_csv("../PerfectGuide/1장/titanic/train.csv")
y_titanic_df = new_titanic_df['Survived']
X_titanic_df = new_titanic_df.drop('Survived', axis=1)

print('\n### y of titanic data ###\n', y_titanic_df.head().to_string())
print('\n### X of titanic data before Preprocessing ###\n', X_titanic_df.head().to_string())
X_titanic_df = transform_features(X_titanic_df)
print('\n### X of titanic data after Preprocessing ###\n', X_titanic_df.head().to_string())

X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size=0.2, random_state=11)

# DecisionTreeClassifier Train/Predict/Evaluate
dt_clf = DecisionTreeClassifier(random_state=11)
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
print('\nDecisionTreeClassifier Accuracy : {0:.4f}'.format(accuracy_score(y_test, dt_pred)))

# RandomForestClassifier Train/Predict/Evaluate
rf_clf = RandomForestClassifier(random_state=11)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
print('\nRandomForestClassifier Accuracy : {0:.4f}'.format(accuracy_score(y_test, rf_pred)))

# LogisticRegression Train/Predict/Evaluate
lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train, y_train)
lr_pred = lr_clf.predict(X_test)
print('\nLogisticRegression Accuracy : {0:.4f}\n'.format(accuracy_score(y_test, lr_pred)))

### ************************************************************************************************************************
# Define a function that processes k-Fold Cross Validation
def exec_kfold(clf, folds=5):
    kfold = KFold(n_splits=folds)
    accuracies = []

    for iter_count, (train_index, test_index) in enumerate(kfold.split(X_titanic_df)):
        X_train, X_test = X_titanic_df.values[train_index], X_titanic_df.values[test_index]
        y_train, y_test = y_titanic_df.values[train_index], y_titanic_df.values[test_index]

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        accu = accuracy_score(y_test, pred)
        print('{0}-th Cross Validation Accuracy Report : {1:.4f}'.format(iter_count, accu))
        accuracies.append(accu)

    mean_accuracy = np.mean(accuracies)
    print('Average Accuracy : {0:.4f}\n'.format(mean_accuracy))


exec_kfold(dt_clf, folds=5)

### ************************************************************************************************************************
scores = cross_val_score(dt_clf, X_titanic_df, y_titanic_df, cv=5)
for iter_count, accu in enumerate(scores):
    print('{0}-th Cross Validation Accuracy Report : {1:.4f}'.format(iter_count, accu))

print('Average Accuracy : {0:.4f}\n'.format(np.mean(scores)))

### ************************************************************************************************************************
parameters = {
    'max_depth': [2, 3, 5, 10],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 5, 8]
}

grid_dclf = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy', cv=5)
grid_dclf.fit(X_train, y_train)

print('GridSearchCV Optimized Hyper Parameter :', grid_dclf.best_params_)
print('GridSearchCV Highest Accuracy : {0:.4f}'.format(grid_dclf.best_score_))

best_dclf = grid_dclf.best_estimator_
dprediction = best_dclf.predict(X_test)
accuracy = accuracy_score(y_test, dprediction)

print('DecisionTreeClassifier Accuracy of Test set : {0:.4f}'.format(accuracy))