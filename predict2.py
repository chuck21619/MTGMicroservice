# routes/predict.py
from fastapi import APIRouter, Request
import psycopg2
import tensorflow as tf
import pickle
import numpy as np
import os
from dotenv import load_dotenv
import os

load_dotenv()  # loads from .env if present
connection_string = os.getenv("DATABASE_URL")

router = APIRouter()
num_slots = 4

@router.post("/predict2")
async def predict2(request: Request):
    data = await request.json()
    username = data.get("username")
    selections = data.get("selections", [])
    game_input_dict = {s['player']: s['deck'] for s in selections}
    print(game_input_dict)
    game_input_dict = {p: d for p, d in game_input_dict.items() if d != "none"}
    game_input_dict = dict(list(game_input_dict.items())[:4])
    try:
        model, players, decks = load_model_from_db(username)
    except Exception as e:
        return {"error": str(e)}
    predicted_winner, preds = predict_winner(model, players, decks, game_input_dict)
    pred_probs = {players[i]: float(preds[i]) for i in range(len(players))}
    return {
        "prediction": predicted_winner,
        "probabilities": pred_probs
    }

def load_model_from_db(username: str):
    conn = psycopg2.connect(connection_string)
    cur = conn.cursor()
    cur.execute("SELECT tf_model, tf_players, tf_decks FROM users WHERE username = %s", (username,))
    row = cur.fetchone()
    if row is None:
        raise Exception(f"No model found for user {username}")
    model_bytes, players_bytes, decks_bytes = row
    players = pickle.loads(players_bytes)
    decks = pickle.loads(decks_bytes)
    import tempfile
    tmp_file = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
    try:
        tmp_file.write(model_bytes)
        tmp_file.flush()
        tmp_file.close()  # Important to close before loading
        model = tf.keras.models.load_model(tmp_file.name)
    finally:
        os.unlink(tmp_file.name)  # Delete the file manually
    
    cur.close()
    conn.close()
    return model, players, decks

def encode_game_input(game_input_dict, players, decks):
    game = dict(game_input_dict)  # copy
    count = 0
    while len(game) < num_slots:
        key = 'NONE' if count == 0 else f'NONE{count}'
        game[key] = 'NONE'
        count += 1
    player_indices = [players.index(p) if p in players else players.index('NONE') for p in game.keys()]
    deck_indices = [decks.index(d) if d in decks else decks.index('NONE') for d in game.values()]
    player_tensor = tf.constant([player_indices], dtype=tf.int32)
    deck_tensor = tf.constant([deck_indices], dtype=tf.int32)
    return player_tensor, deck_tensor

def predict_winner(model, players, decks, game_input_dict):
    player_tensor, deck_tensor = encode_game_input(game_input_dict, players, decks)
    preds = model.predict([player_tensor, deck_tensor])[0]
    max_idx = np.argmax(preds)
    predicted_winner = players[max_idx]
    return predicted_winner, preds