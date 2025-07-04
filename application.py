from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


app = Flask(__name__)

@app.route("/", methods=["GET"])
def home_page():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict_datapoint():
    if request.method == "POST":
        try:
            data = CustomData(
                gender=request.form.get("gender"),
                race_ethnicity=request.form.get("ethnicity"),
                parental_level_of_education=request.form.get("parental_level_of_education"),
                lunch=request.form.get("lunch"),
                test_preparation_course=request.form.get("test_preparation_course"),
                reading_score=int(request.form.get("reading_score")),
                writing_score=int(request.form.get("writing_score")),
            )

            input_df = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            result = predict_pipeline.predict(input_df)

            return render_template("home.html", results=result[0])

        except Exception as e:
            return render_template("home.html", results=f"Terjadi error: {e}")

    return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
