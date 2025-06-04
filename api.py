from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is live"}

@app.post("/predict")
def predict():
    print("predict called")
    return {
        "winner": "chuck"
    }