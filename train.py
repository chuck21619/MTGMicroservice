from fastapi import APIRouter, Request
import psycopg2
from data_generation import *
from preprocessing import *
from model import *
import pickle

router = APIRouter()

@router.post("/train")
async def train(request: Request):
    data = await request.json()
    url = data.get("url")
    
    df = generate_dataset(url)

    print("training model")
    x_player, y_player, le_input_players, le_target_players, x_deck, y_deck, le_input_decks, le_target_decks = encode_data(df)
    model, combined_features = train_model(x_player, y_player, le_input_players, le_target_players, x_deck, y_deck, le_input_decks, le_target_decks)

    print("connecting to database")
    #connecting to database
    connection = psycopg2.connect("postgresql://postgres:notastupidpassword@localhost:5432/my_local_db?sslmode=disable")
    current = connection.cursor()

    print("saving model to database")
    #save model to database
    model_bytes = pickle.dumps(model)
    current.execute("UPDATE users SET decision_tree = %s WHERE id = %s", (psycopg2.Binary(model_bytes), 3))
    connection.commit()
    current.close()
    connection.close()
    return {"awef": "awef"}
