import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score


train = pd.read_csv("../../data/titanic/train.csv")
test = pd.read_csv("../../data/titanic/test.csv")
sub_df = pd.read_csv("../../data/titanic/sample_submission.csv")

df = pd.concat([train, test])

# Cabin split -> Deck, Number, Side
df[['Cabin_Deck', 'Cabin_Number', 'Cabin_Side']] = df['Cabin'].str.split('/', expand=True)
df = df.drop(['Cabin', 'Cabin_Number'], axis=1)

df = df.drop(['PassengerId', 'Name'], axis=1)

# 범주형 칼럼 선택, y 값 빼두기
categorical_cols = df.columns[(df.dtypes == object) & (df.columns != 'Transported')]

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 결측치 처리 -> 다시해보기
df.fillna(-99, inplace=True)

train_n = len(train)
train = df.iloc[:train_n, :]
test = df.iloc[train_n:, :]

y = train['Transported'].astype('bool')

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)],
    remainder='passthrough'
)

X_train = preprocessor.fit_transform(train.drop(['Transported'], axis=1))
X_test = preprocessor.transform(test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

base_models = [
    ('logistic', LogisticRegression(max_iter=1000)),
    ('decision_tree', DecisionTreeClassifier()),
    ('random_forest', RandomForestClassifier()),
    ('gbrboost', GradientBoostingClassifier())
]

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression()  # 메타 모델
)

param_grid = {
    'random_forest__n_estimators': [50, 100],
    'random_forest__max_depth': [None, 10, 20],
    'gbrboost__n_estimators': [100, 200],
    'gbrboost__learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(
    estimator=stacking_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y)


predictions = grid_search.predict(X_test_scaled)

sub_df["Transported"] = predictions
sub_df['Transported'] = sub_df['Transported'].astype(bool)

sub_df.to_csv("../../data/titanic/sample_submission_dummies3.csv", index=False)

sub_df.head()
