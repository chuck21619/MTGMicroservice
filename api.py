from fastapi import FastAPI
from fastapi import Request

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is live"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    url = data.get("url")
    print(f"predict called with URL: {url}")
    return {
        "winner": "chuck"
    }