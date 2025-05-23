
# import libraries which are used in the code
import pandas as pd
import mlflow


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import  OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


class SmartPremiumPipeline:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.fitted = False

    def _clean_data(self, df):
        if 'Policy Start Date' in df.columns:
            df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'], errors='coerce')
            df['Policy_Year'] = df['Policy Start Date'].dt.year
            df['Policy_Month'] = df['Policy Start Date'].dt.month
            df['Policy_Weekday'] = df['Policy Start Date'].dt.weekday

        if 'Age' in df.columns:
            df['Age Group'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 60, 100],
                                     labels=['18-25', '26-35', '36-45', '46-60', '60+'])

        if 'Credit Score' in df.columns:
            df['CreditScoreGroup'] = pd.cut(df['Credit Score'], bins=[300, 500, 650, 750, 850],
                                            labels=['Poor', 'Average', 'Good', 'Excellent'])

        return df

    def _remove_outliers(self, df, numerical_cols):
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    def fit(self, df, target_column, model_name='RandomForest', model_params={}):
        df = df.copy()

        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        for col in num_cols:
            df[col].fillna(df[col].median(), inplace=True)
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)

        df.drop_duplicates(inplace=True)

        if target_column in num_cols:
            num_cols.remove(target_column)
        df = self._remove_outliers(df, num_cols)

        df = self._clean_data(df)

        cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_features = df.select_dtypes(include=['int64', 'float64']).drop(columns=[target_column]).columns.tolist()

        self.preprocessor = ColumnTransformer([
            ('num', 'passthrough', num_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
        ])

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #with mlflow.start_run():
        if model_name == 'RandomForest':
            model = RandomForestRegressor(random_state=42, **model_params)
        elif model_name == 'XGBoost':
            model = XGBRegressor(random_state=42, **model_params)
        elif model_name == 'LinearRegression':
            model = LinearRegression(**model_params)
        elif model_name == 'DecisionTree':
            model = DecisionTreeRegressor(random_state=42, **model_params)
        else:
            raise ValueError("Unsupported model")

        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', model)
        ])
        with mlflow.start_run(run_name=f"{model_name}_Run"):
            self.model.fit(X_train, y_train)
            self.fitted = True

            y_pred = self.model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print("Model Evaluation:", model_name)
            print("RMSE:", rmse)
            print("MAE:", mae)
            print("R^2 Score:", r2)

            mlflow.log_param("model_name", model_name)
            mlflow.log_params(model_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            mlflow.sklearn.log_model(self.model, artifact_path=f"model_{model_name}")

        

    def predict(self, df):
        if not self.fitted:
            raise Exception("Model not fitted. Please call fit() first.")

        df = df.copy()

        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            df[col].fillna(df[col].median(), inplace=True)
        for col in df.select_dtypes(include=['object', 'category', 'bool']).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

        df = self._clean_data(df)
        
        cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        self.preprocessor = ColumnTransformer([
            ('num', 'passthrough', num_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
        ])

        predictions = self.model.predict(df)
        return predictions

