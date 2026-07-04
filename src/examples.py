from src.models.custom.custom_classifier_v1 import Custom_Classifier


# Load model
custom_model = Custom_Classifier()
custom_model.load_model(name='custom_classifier_model_v1.2.pkl')

sentence1 = "I went into this movie with low expectations, but it completely surprised me. The story was engaging, the performances felt genuine, and the ending was satisfying without being predictable. It wasn't perfect—the pacing slowed down in the middle—but overall I had a great time and would happily watch it again."
sentence2 = "The acting was excellent and the cinematography was beautiful, but the story was painfully slow and by the end I couldn't wait for it to be over. I appreciate what the director was trying to do, but I wouldn't recommend this movie."
print(custom_model.predict(sentence1,True))
print(custom_model.predict(sentence2,True))