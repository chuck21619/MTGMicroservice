from fastapi import APIRouter, Request
import pandas as pd
import requests
from io import StringIO
import tensorflow as tf

router = APIRouter()

@router.post("/train2")
async def train2(request: Request):
    data = await request.json()
    url = data.get("url")
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": f"Failed to fetch CSV. Status code: {response.status_code}"}

    df = pd.read_csv(StringIO(response.text))
    print("Data preview:")

    cleaned_games = df.drop(columns=['winner'])
    winners = df['winner'].tolist()

    # Convert DataFrame to list of dicts for padding like in JS
    games = cleaned_games.where(pd.notnull(cleaned_games), None).to_dict(orient="records")

    # Pad each game dict to have at least 4 keys with keys NONE, NONE1, NONE2...
    for game in games:
        count = 0
        while len(game) < 4:
            key = 'NONE' if count == 0 else f'NONE{count}'
            game[key] = 'NONE'
            count += 1
            
    #inputTensors
    #createModel
    #trainModel
    #save model
    
    return {
        "winners": winners,
        "paddedGames": games
    }
