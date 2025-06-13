import numpy as np
import xgboost as xgb
from preprocessing import encode_game_input

model_player = None
model_deck = None
model_meta = None
le_input_players = None
le_target_players = None
le_input_decks = None
le_target_decks = None


def post_filter_prediction(pred_probs, allowed_labels, label_encoder):
    classes = label_encoder.inverse_transform(np.arange(pred_probs.shape[0]))
    
    for idx, label in enumerate(classes):
        if label not in allowed_labels:
            pred_probs[idx] = 0
    
    # Renormalize (optional but good practice)
    if pred_probs.sum() > 0:
        pred_probs = pred_probs / pred_probs.sum()
    
    return np.argmax(pred_probs)


def train_model(x_player, y_player, le_input_players_input, le_target_players_input, x_deck, y_deck, le_input_decks_input, le_target_decks_input):
    global model_meta, model_player, model_deck, le_input_players, le_target_players, le_input_decks, le_target_decks

    learning_rate = 0.1
    colsample_bytree = 0.6
    subsample = 0.6
    n_estimators = 250
    max_depth = 2

    le_input_players = le_input_players_input
    le_target_players = le_target_players_input
    le_input_decks = le_input_decks_input
    le_target_decks = le_target_decks_input

    model_player = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, colsample_bytree=colsample_bytree, subsample=subsample)
    model_player.fit(x_player, y_player)

    model_deck = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, colsample_bytree=colsample_bytree, subsample=subsample)
    model_deck.fit(x_deck, y_deck)

    deck_probs = model_deck.predict_proba(x_deck)
    deck_filtered_preds = np.zeros(deck_probs.shape[0], dtype=int)

    for i in range(x_deck.shape[0]):
        allowed_decks = set(le_input_decks.inverse_transform(x_deck[i]))
        deck_filtered_preds[i] = post_filter_prediction(deck_probs[i], allowed_decks, le_target_decks)

    player_probs = model_player.predict_proba(x_player)
    player_filtered_preds = np.zeros(player_probs.shape[0], dtype=int)

    for i in range(x_player.shape[0]):
        allowed_players = set(le_input_players.inverse_transform(x_player[i]))
        player_filtered_preds[i] = post_filter_prediction(player_probs[i], allowed_players, le_target_players)

    combined_features = np.vstack((player_filtered_preds, deck_filtered_preds)).T
    model_meta = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, colsample_bytree=colsample_bytree, subsample=subsample)
    model_meta.fit(combined_features, y_player)

    return model_meta, model_player, model_deck


def model_predict(game_input):
    encoded_game_input = encode_game_input(game_input, le_input_players, le_input_decks)

    x_player_input = encoded_game_input[:, 0].reshape(1, -1)
    x_deck_input = encoded_game_input[:, 1].reshape(1, -1)

    player_probs = model_player.predict_proba(x_player_input)
    deck_probs = model_deck.predict_proba(x_deck_input)

    allowed_players = set(game_input.keys())
    allowed_decks = set(game_input.values())

    player_filtered_pred = post_filter_prediction(player_probs[0], allowed_players, le_target_players)
    deck_filtered_pred = post_filter_prediction(deck_probs[0], allowed_decks, le_target_decks)
    combined_features = np.array([player_filtered_pred, deck_filtered_pred]).reshape(1, -1)

    final_prediction = model_meta.predict(combined_features)

    predicted_winner = le_target_players.inverse_transform(final_prediction)
    return predicted_winner[0]
