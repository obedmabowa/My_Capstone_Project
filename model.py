import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def train_model():
    df = pd.read_csv("Avocado_Prices_Data.csv")

    df = df.dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year

    df = pd.get_dummies(df, columns=["region"], drop_first=True)

    X = df.drop(["AveragePrice", "Date"], axis=1)
    y = df["AveragePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    return model, X.columns
