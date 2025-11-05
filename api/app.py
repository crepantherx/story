from fastapi import FastAPI, Depends, HTTPException, status, Request
app = FastAPI()

def authenticate(request: Request):

    password = request.headers.get("Authorization")

    password = password.split(" ")[-1]

    if password != "sudhir@123":
        raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not Authenticated",
                headers={"WWW-Authenticate": "Basic"}
            )

@app.get("/")
def home():
    return {"status": "sudhir"}

@app.get("/status")
def abc(_: bool = Depends(authenticate)):
    ...
    return {"ok": "sudhir"}
