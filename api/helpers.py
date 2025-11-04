from fastapi import FastAPI, Request, Depends, HTTPException, status

MODELS = {
        "models": {
            'classification': {
                "tree": {
                    "RandomForestClassifier": {'last_trained_on': '2023'},
                    "DecisionTreeClassifier": {'last_trained_on': '2023'}
                }
            },
            "regression": {
            }
        }
    }

TOKEN = "supersecrettoken123"
def authenticate(request: Request):
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not Authenticated",
            headers={"WWW-Authenticate": "Basic"}
        )

    token = auth_header.split(" ")[1]
    if token != TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not Authenticated",
            headers={"WWW-Authenticate": "Basic"}
        )

    return True


from pydantic import BaseModel
class ModifyRequest(BaseModel):
    a: int
    b: int
    c: int
    d: int
