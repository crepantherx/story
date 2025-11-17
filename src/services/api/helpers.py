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