# routes/predict.py
from fastapi import APIRouter, Request

router = APIRouter()

@router.post("/predict2")
async def predict2(request: Request):
    
    data = await request.json()
    selections = data.get("selections", [])
    game_input_dict = {s['player']: s['deck'] for s in selections}
    print(game_input_dict)
    return {"prediction": "Kakarot"}
