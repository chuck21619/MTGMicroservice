from fastapi import FastAPI, Request
from data_generation import *
from preprocessing import *
from model import *
import psycopg2
import pickle
from train import router as train_router
import model
import binascii


app = FastAPI()
app.include_router(train_router)

@app.get("/")
def read_root():
    return {"message": "API is live"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    selections = data.get("selections", [])
    game_input_dict = {s['player']: s['deck'] for s in selections}
    print(game_input_dict)

    userID = 3

    #connecting to database
    connection = psycopg2.connect("postgresql://postgres:notastupidpassword@localhost:5432/my_local_db?sslmode=disable")
    current = connection.cursor()

    # get models and encoders from database
    current.execute("""
        SELECT model_meta, model_player, model_deck, 
            le_input_players, le_target_players, 
            le_input_decks, le_target_decks 
        FROM users 
        WHERE id = %s
    """, (userID,))

    row = current.fetchone()
    if row:
        model_meta = pickle.loads(binascii.unhexlify(row[0][2:])) if row[0] else None
        model_player = pickle.loads(binascii.unhexlify(row[1][2:])) if row[1] else None
        model_deck = pickle.loads(binascii.unhexlify(row[2][2:])) if row[2] else None
        le_input_players = pickle.loads(binascii.unhexlify(row[3][2:])) if row[3] else None
        le_target_players = pickle.loads(binascii.unhexlify(row[4][2:])) if row[4] else None
        le_input_decks = pickle.loads(binascii.unhexlify(row[5][2:])) if row[5] else None
        le_target_decks = pickle.loads(binascii.unhexlify(row[6][2:])) if row[6] else None


        print("Models and encoders loaded successfully.")
    else:
        print("No models found for user.")

    current.close()
    connection.close()


    model.model_meta = model_meta
    model.model_deck = model_deck
    model.model_player = model_player
    model.le_input_decks = le_input_decks
    model.le_input_players = le_input_players
    model.le_target_decks = le_target_decks
    model.le_target_players = le_target_players
    winner = model_predict(game_input_dict)

    return {
        "prediction": winner,
    }