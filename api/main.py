from fastapi import FastAPI, Request
import pickle
from sklearn import datasets
import numpy as np
import mlflow
from api.scratch import XGBoostClassifier
from mlflow.tracking import MlflowClient
dataset = datasets.load_iris()

app = FastAPI()

import mlflow.pyfunc

class MyCustomXGBoostWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, custom_model):
        self.custom_model = custom_model

    def predict(self, context, model_input):
        # Convert pandas DataFrame to np.array for prediction
        import numpy as np
        if hasattr(model_input, "values"):
            arr = model_input.values  # DataFrame -> np.ndarray
        else:
            arr = np.array(model_input)
        return self.custom_model.predict(arr)


def load_production_model():
    import mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")  # Set explicitly!
    # model_uri = "models:/crepantherx/Production"  # “Production” stage, always use latest prod model
    model = mlflow.pyfunc.load_model("models:/crepantherx/Production")
    return model

@app.get("/")
def home():
    return {"status": "ok"}

@app.get("/predict")
def predict(a,b,c,d):
    a, b, c, d = map(float, (a, b, c, d))
    model = load_production_model()
    import numpy as np
    res = str(dataset.target_names[int(model.predict(np.array([[a, b, c, d]]))[0])])
    return {"prediction": res}

@app.post("/train")
async def train(request: Request):
    data = await request.json()
    X = np.array(data["X"])
    y = np.array(data["y"])
    clf = XGBoostClassifier(n_estimators=5, max_depth=2)
    clf.fit(X, y)

    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("cycle2")

    mlflow.pyfunc.log_model(
        artifact_path="klm",
        python_model=MyCustomXGBoostWrapper(clf),
        registered_model_name="crepantherx"
    )

    client = MlflowClient()
    model_name = "crepantherx"
    versions = client.get_latest_versions(model_name, stages=["None"])
    if versions:
        latest_ver = versions[0].version
        client.transition_model_version_stage(model_name, latest_ver, "Production")

    return {"status": "Model trained", "shape": X.shape}
