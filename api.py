from fastapi import FastAPI
from fastapi import Request
from data_generation import *
from preprocessing import *
from model import *
from sklearn.metrics import accuracy_score

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is live"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    url = data.get("url")
    print(f"predict called with URL: {url}")
    
    df = generate_dataset(url)
    print("df")
    print(df)

    x_player, y_player, le_input_players, le_target_players, x_deck, y_deck, le_input_decks, le_target_decks = encode_data(df)
    model, combined_features = train_model(x_player, y_player, le_input_players, le_target_players, x_deck, y_deck, le_input_decks, le_target_decks)
    meta_predictions = model.predict(combined_features)
    accuracy = accuracy_score(y_player, meta_predictions)
    print("accuracy:", accuracy)

    return {
        "prediction": "victor",
        "accuracy": accuracy
    }