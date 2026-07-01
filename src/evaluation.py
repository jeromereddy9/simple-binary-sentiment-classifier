import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.models.custom.custom_classifier import Custom_Classifier
from src.utils import path_builder

# Load models
custom_model = Custom_Classifier()
custom_model.load_model(name='custom_classifier_model_v1.2.pkl')
custom_model.get_model_summary()

baseline_model_path = path_builder('src/models/saved_models/sentiment_classifier.keras')
baseline_model = load_model(baseline_model_path)

# Data
x_test, y_test = custom_model.get_test_split()

print("Evaluation Starting:\n")

# custom model (string labels)
y_pred_custom = np.array([
    custom_model.predict(x) for x in x_test
])

print("CUSTOM MODEL RESULTS")
print(f"Accuracy: {(accuracy_score(y_test, y_pred_custom)*100):.2f}%")
print(classification_report(y_test, y_pred_custom))
print(confusion_matrix(y_test, y_pred_custom))

# Baseline (numeric labels)
label_map = {"negative": 0, "positive": 1}
inv_map = {0: "negative", 1: "positive"}

# convert ground truth
y_test_int = np.array([label_map[y] for y in y_test])

# predictions
y_pred_baseline_probs = baseline_model.predict(x_test, verbose=0)
y_pred_baseline = (y_pred_baseline_probs > 0.5).astype(int).flatten()

print("\nBASELINE MODEL RESULTS")
print(f"Accuracy: {(accuracy_score(y_test_int, y_pred_baseline)*100):.2f}%")
print(classification_report(y_test_int, y_pred_baseline))
print(confusion_matrix(y_test_int, y_pred_baseline))