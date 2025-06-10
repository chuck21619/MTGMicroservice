from fastapi import FastAPI
from fastapi import Request
from data_generation import *
from preprocessing import *
from model import *
from sklearn.metrics import accuracy_score
import psycopg2
import pickle


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is live"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    url = data.get("url")
    print(f"predict called with URL: {url}")
    
    df = generate_dataset(url)
    print("df")
    print(df)

    x_player, y_player, le_input_players, le_target_players, x_deck, y_deck, le_input_decks, le_target_decks = encode_data(df)
    model, combined_features = train_model(x_player, y_player, le_input_players, le_target_players, x_deck, y_deck, le_input_decks, le_target_decks)

    #connecting to database
    connection = psycopg2.connect("postgresql://postgres:notastupidpassword@localhost:5432/my_local_db?sslmode=disable")
    current = connection.cursor()

    #save model to database
    # model_bytes = pickle.dumps(model)
    # current.execute("UPDATE users SET decision_tree = %s WHERE id = %s", (psycopg2.Binary(model_bytes), 3))
    # connection.commit()
    # current.close()
    # connection.close()


    #get model from database
    current.execute("SELECT decision_tree FROM users WHERE id = %s", (3,))
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

    ## test connection to database. just read some field from it
    ## so goose migrations to add columns for decision_tree and neural_network

    #get database connection string from environment variables
    #save model to database

    meta_predictions = clf.predict(combined_features)
    accuracy = accuracy_score(y_player, meta_predictions)
    print("accuracy:", accuracy)

    return {
        "prediction": "victor",
        "accuracy": accuracy
    }