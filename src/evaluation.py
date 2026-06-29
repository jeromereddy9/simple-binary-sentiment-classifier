import os, sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.models.custom.custom_classifier import Custom_Classifier
from sklearn.metrics import classification_report,confusion_matrix

model = Custom_Classifier()
model.load_model(name='custom_classifier_model_v1.2.pkl')
model.get_model_summary()


x_test, y_test = model.get_test_split()

correct = 0
y_predictions = []
print("Evaluation Starting:")
for sentence,label in zip(x_test,y_test):
    prediction = model.predict(sentence)
    y_predictions.append(prediction)
    if prediction == label:
        correct += 1



print(f"\nAccuracy: {(correct/len(y_test))*100:.2f}%\n")
print(classification_report(y_test,y_predictions))
print(confusion_matrix(y_test,y_predictions))
