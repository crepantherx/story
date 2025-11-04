from api.helpers import *
from fastapi import FastAPI, Request, Depends, HTTPException, status
app = FastAPI()

@app.get("/models")
def models(_: bool = Depends(authenticate)):
    return MODELS

@app.get("/models/{model_type}/{model_cat}/{model_name}")
def model_info(model_type: str, model_cat: str, model_name: str, request: Request, _: bool = Depends(authenticate)):
    return MODELS["models"][model_type][model_cat][model_name]