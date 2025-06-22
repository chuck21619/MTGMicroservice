from fastapi import APIRouter, Request
import pandas as pd
import requests
from io import StringIO
import tensorflow as tf
import tempfile
import psycopg2
import pickle
import os

router = APIRouter()
num_slots = 4

@router.post("/train2")
async def train2(request: Request):
    data = await request.json()
    url = data.get("url")
    username = data.get("username")
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": f"Failed to fetch CSV. Status code: {response.status_code}"}

    df = pd.read_csv(StringIO(response.text))
    print("Data preview:")

    cleaned_games = df.drop(columns=['winner'])
    winners = df['winner'].tolist()

    # Convert DataFrame to list of dicts like your JS, filtering out None and NaN
    games = cleaned_games.to_dict(orient="records")
    games = [{k: v for k, v in game.items() if v == v and v is not None} for game in games]

    # Pad each game dict to have at least 4 keys, using keys like NONE, NONE1, NONE2
    for game in games:
        count = 0
        while len(game) < num_slots:
            key = 'NONE' if count == 0 else f'NONE{count}'
            game[key] = 'NONE'
            count += 1

    # Convert cleaned_games DataFrame to list of dicts for extracting unique players and decks
    cleaned_games_list = cleaned_games.to_dict(orient="records")

    players = sorted({player for game in cleaned_games_list for player in game.keys()})
    decks = sorted({deck for game in cleaned_games_list for deck in game.values() if isinstance(deck, str)})

    # Add 'NONE' to players and decks so padding keys don't cause errors
    if 'NONE' not in players:
        players.append('NONE')
    if 'NONE' not in decks:
        decks.append('NONE')

    player_data = []
    deck_data = []

    for game in games:  # games is padded already
        player_row = []
        deck_row = []
        for player, deck in game.items():
            player_row.append(players.index(player))
            deck_row.append(decks.index(deck))
        player_data.append(player_row)
        deck_data.append(deck_row)

    player_tensor = tf.constant(player_data, dtype=tf.int32)
    deck_tensor = tf.constant(deck_data, dtype=tf.int32)

    # Inputs
    player_input = tf.keras.layers.Input(shape=(num_slots,), dtype='int32', name='playerInput')
    deck_input = tf.keras.layers.Input(shape=(num_slots,), dtype='int32', name='deckInput')
    num_players = len(players)
    num_decks = len(decks)

    # Embeddings
    player_embedding = tf.keras.layers.Embedding(
        input_dim=num_players,
        output_dim=4,  # small embedding for players
        name='playerEmbedding'
    )(player_input)

    deck_embedding = tf.keras.layers.Embedding(
        input_dim=num_decks,
        output_dim=8,  # larger embedding for decks
        name='deckEmbedding'
    )(deck_input)

    # Flatten embeddings
    player_flat = tf.keras.layers.Flatten()(player_embedding)
    deck_flat = tf.keras.layers.Flatten()(deck_embedding)

    # Concatenate
    combined = tf.keras.layers.Concatenate()([player_flat, deck_flat])

    # Dense layers
    dense1 = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(combined)

    # Output
    output = tf.keras.layers.Dense(num_players, activation='softmax', name='output')(dense1)

    # Model
    model = tf.keras.Model(inputs=[player_input, deck_input], outputs=output, name='PlayerDeckModel')

    # Compile
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    encoded_winners = [players.index(winner) for winner in winners]
    target_tensor = tf.constant(encoded_winners, dtype=tf.int32)

    model.fit([player_tensor, deck_tensor], target_tensor, epochs=10)

    # Save model as single .keras file in temp directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_path = os.path.join(tmpdirname, "tf_model.keras")
        model.save(model_path)  # Saves as single file .keras by default

        # Read the .keras file as bytes
        with open(model_path, "rb") as f:
            model_bytes = f.read()

    # Pickle players and decks lists
    players_bytes = pickle.dumps(players)
    decks_bytes = pickle.dumps(decks)

    # Save to PostgreSQL
    conn = psycopg2.connect("postgresql://postgres:notastupidpassword@localhost:5432/my_local_db?sslmode=disable")
    cur = conn.cursor()

    cur.execute("""
        UPDATE users SET 
            tf_model = %s,
            tf_players = %s,
            tf_decks = %s
        WHERE username = %s
    """, (
        psycopg2.Binary(model_bytes),
        psycopg2.Binary(players_bytes),
        psycopg2.Binary(decks_bytes),
        username
    ))

    conn.commit()
    cur.close()
    conn.close()

    return {
        "result": "trained and saved tensorflow model"
    }
