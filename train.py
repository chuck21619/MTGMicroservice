from fastapi import APIRouter, Request
import psycopg2
from data_generation import *
from preprocessing import *
from model import *
import pickle
from dotenv import load_dotenv
import os

load_dotenv()  # loads from .env if present
connection_string = os.getenv("DATABASE_URL")

router = APIRouter()

@router.post("/train")
async def train(request: Request):
    data = await request.json()
    url = data.get("url")
    username = data.get("username")
    
    df = generate_dataset(url)

    print("training model")
    x_player, y_player, le_input_players, le_target_players, x_deck, y_deck, le_input_decks, le_target_decks = encode_data(df)
    model_meta, model_player, model_deck = train_model(x_player, y_player, le_input_players, le_target_players, x_deck, y_deck, le_input_decks, le_target_decks)

    print("connecting to database")
    #connecting to database
    connection = psycopg2.connect(connection_string)
    current = connection.cursor()

    print("saving model to database")
    # serialize models and encoders
    model_meta_bytes = pickle.dumps(model_meta)
    model_player_bytes = pickle.dumps(model_player)
    model_deck_bytes = pickle.dumps(model_deck)
    le_input_players_bytes = pickle.dumps(le_input_players)
    le_target_players_bytes = pickle.dumps(le_target_players)
    le_input_decks_bytes = pickle.dumps(le_input_decks)
    le_target_decks_bytes = pickle.dumps(le_target_decks)

    current.execute("""
        UPDATE users SET 
            model_meta = %s,
            model_player = %s,
            model_deck = %s,
            le_input_players = %s,
            le_target_players = %s,
            le_input_decks = %s,
            le_target_decks = %s
        WHERE username = %s
    """, (
        psycopg2.Binary(model_meta_bytes),
        psycopg2.Binary(model_player_bytes),
        psycopg2.Binary(model_deck_bytes),
        psycopg2.Binary(le_input_players_bytes),
        psycopg2.Binary(le_target_players_bytes),
        psycopg2.Binary(le_input_decks_bytes),
        psycopg2.Binary(le_target_decks_bytes),
        username
    ))


    connection.commit()
    current.close()
    connection.close()
    return {"status": "model and encoders saved"}
