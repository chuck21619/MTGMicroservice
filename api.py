from fastapi import FastAPI, Request
from data_generation import *
from preprocessing import *
from model import *
from sklearn.metrics import accuracy_score
import psycopg2
import pickle
from train import router as train_router


app = FastAPI()
app.include_router(train_router)

@app.get("/")
def read_root():
    return {"message": "API is live"}

@app.post("/predict")
async def predict(request: Request):
    print("TIME TO BE ALIVE ==============================")
    data = await request.json()
    print(data)


    return {
        "prediction": "victor",
        "accuracy": ""
    }


    userID = 3

    #connecting to database
    connection = psycopg2.connect("postgresql://postgres:notastupidpassword@localhost:5432/my_local_db?sslmode=disable")
    current = connection.cursor()

    #get model from database
    current.execute("SELECT decision_tree FROM users WHERE id = %s", (userID,))
    row = current.fetchone()
    if row and row[0]:
        model_bytes = row[0]
        clf = pickle.loads(model_bytes)
        print("Model loaded successfully.")

    current.execute("SELECT email FROM users WHERE id = 4")
    result = current.fetchone()
    if result:
        print("email: ", result[0])
    else:
        print("shit fucked up")

    current.close()
    connection.close()

    meta_predictions = clf.predict(combined_features)
    accuracy = accuracy_score(y_player, meta_predictions)
    print("accuracy:", accuracy)

    return {
        "prediction": "victor",
        "accuracy": accuracy
    }