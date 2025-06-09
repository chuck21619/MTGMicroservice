import pandas as pd

def build_dataset_from_sheet(df):
    games = []
    for _, row in df.iterrows():
        game = {}
        for player in row.index:
            if player != "winner" and pd.notna(row[player]) and row[player] != '':
                game[player] = row[player]
        game["winner"] = row["winner"]
        games.append(game)
    return pd.DataFrame(games)

def generate_dataset(url):
    raw_data = pd.read_csv(url)
    return build_dataset_from_sheet(raw_data)