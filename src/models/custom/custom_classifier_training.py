from src.models.custom.custom_classifier_v1 import Custom_Classifier
from src.preprocessing import Preprocessor



path = 'data/IMDB Dataset.csv'
p = Preprocessor(path,128)
model = Custom_Classifier(p,64,0.2)
model.fit(100,name='custom_classifier_model_v1.2.pkl',decay_rate=0.99)