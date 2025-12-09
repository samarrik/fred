from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health_check():
    """
    Simple endpoint to check that the server is running.
    When you visit /health in the browser, it should return {"status": "ok"}.
    """
    return {"status": "ok"}


@app.get("/")
def read_root():
    """
    Root endpoint.
    Visiting http://localhost:8000/ should show this JSON.
    """
    return {"message": "Hello from avatar_mvp backend!"}
