import joblib
import pandas as pd

class RandomForestClassifier:
    # the constructor which loads preprocessing objects and Random Forest object (created with Jupyter notebook)
    def __init__(self):
        path_to_artifacts = "../../research/"
        self.values_fill_missing =  joblib.load(path_to_artifacts + "train_mode.joblib")
        self.encoders = joblib.load(path_to_artifacts + "encoders.joblib")
        self.model = joblib.load(path_to_artifacts + "random_forest.joblib")
    # the method which takes as input JSON data, converts it to Pandas DataFrame and apply pre-processing
    def preprocessing(self, input_data):
        # JSON to pandas DataFrame
        input_data = pd.DataFrame(input_data, index=[0])
        # fill missing values
        input_data.fillna(self.values_fill_missing)
        # convert categoricals
        for column in [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]:
            categorical_convert = self.encoders[column]
            input_data[column] = categorical_convert.transform(input_data[column])

        return input_data
    # the method that calls ML for computing predictions on prepared data,
    def predict(self, input_data):
        return self.model.predict_proba(input_data)
    # the method that applies post-processing on prediction values,
    def postprocessing(self, input_data):
        label = "<=50K"
        if input_data[1] > 0.5:
            label = ">50K"
        return {"probability": input_data[1], "label": label, "status": "OK"}
    # the method that combines: preprocessing, predict and postprocessing and returns JSON object with the response.
    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)[0]  # only one sample
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction