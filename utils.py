# utils.py
# Shared utility functions and classes for the Employee Productivity Prediction project

from sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns
        self.label_encoders = {}

    def fit_transform(self, data):
        output = data.copy()
        if self.columns is not None:
            for col in self.columns:
                self.label_encoders[col] = LabelEncoder()
                output[col] = self.label_encoders[col].fit_transform(output[col])
        return output

    def transform(self, data):
        output = data.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = self.label_encoders[col].transform(output[col])
        return output