from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

print("loading datasets")

data=fetch_california_housing()

X=pd.DataFrame(data.data,columns=data.feature_names)
y=data.target
print(f"total records: {X.shape[0]}")

X_train,X_test, y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

model=RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print(f"MAE: ${mae *100000:,.0f}")

joblib.dump(model,"house_model.joblib")

joblib.dump(list(X.columns),"housea_features.joblib")




