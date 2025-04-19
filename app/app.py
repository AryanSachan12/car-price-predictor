from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
car = pd.read_csv("../model/datasets/cleaned_car.csv")

model = pickle.load(open("../model/model.pkl", "rb"))


@app.route("/")
def index():
    companies = sorted(car["company"].unique())
    car_models = sorted(car["name"].unique())
    years = sorted(car["year"].unique(), reverse=True)
    fuel_type = car["fuel_type"].unique()

    company_to_models = {}
    for company in companies:
        models = car[car["company"] == company]["name"].unique().tolist()
        company_to_models[company] = models

    return render_template(
        "index.html",
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_type=fuel_type,
        company_to_models=company_to_models,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        company = request.form.get("company")
        car_model = request.form.get("car_model")
        year = int(request.form.get("year"))
        fuel_type = request.form.get("fuel_type")
        kms_driven = int(request.form.get("kilo_driven"))

        input_df = pd.DataFrame(
            [[car_model, company, year, kms_driven, fuel_type]],
            columns=["name", "company", "year", "kms_driven", "fuel_type"],
        )

        prediction = model.predict(input_df)

        # Convert the float64 value to a string
        return str(np.round(prediction[0], 2))

    except Exception as e:
        return f"Error: {e}"
