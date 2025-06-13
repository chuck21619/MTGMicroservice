import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_data(df):
    df = df.fillna('none')
    df_input = df.drop(columns="winner")

    # player x and y
    player_names = sorted(df.columns.drop("winner"))
    le_input_players = LabelEncoder()
    le_input_players.fit(player_names + ['none'])

    player_input = np.tile(df_input.columns.to_numpy(), (df_input.shape[0], 1))
    x_player_almost = le_input_players.transform(player_input.ravel()).reshape(player_input.shape)
    none_player_code = le_input_players.transform(["none"])[0]
    x_player = np.where(df_input.values != "none", x_player_almost, none_player_code)

    unique_winner_players = df['winner'].unique()
    le_target_players = LabelEncoder()
    le_target_players.fit(unique_winner_players)
    y_player = le_target_players.transform(df["winner"])

    # deck x and y
    deck_names = sorted(pd.unique(df.drop(columns="winner").values.ravel()))
    le_input_decks = LabelEncoder()
    le_input_decks.fit(deck_names + ['none'])

    x_deck = le_input_decks.transform(df.drop(columns="winner").values.ravel()).reshape(df_input.shape)

    winner_deck_map = df.apply(lambda row: row[row["winner"]], axis=1)
    le_target_decks = LabelEncoder()
    le_target_decks.fit(sorted(winner_deck_map.unique()))
    y_deck = le_target_decks.transform(winner_deck_map)

    return x_player, y_player, le_input_players, le_target_players, x_deck, y_deck, le_input_decks, le_target_decks

def encode_game_input(game_input, le_input_players, le_input_decks):
    """
    Encode the game input (player-deck pairs) using the provided label encoders.
    """
    encoded_input = []
    
    for player, deck in game_input.items():
        player_encoded = le_input_players.transform([player])[0]
        deck_encoded = le_input_decks.transform([deck])[0]
        encoded_input.append([player_encoded, deck_encoded])
    
    return np.array(encoded_input)
