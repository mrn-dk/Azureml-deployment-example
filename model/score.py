import json
import joblib
import numpy as np
import os


def init() -> None:
    global model
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model.pkl"
    )
    model = joblib.load(model_path)


def run(data: str) -> str:
    """Process the input data and return the prediction result."""
    try:
        data = np.array(json.loads(data)["data"])
        result = model.predict(data).tolist()
        return json.dumps({"result": result})
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
