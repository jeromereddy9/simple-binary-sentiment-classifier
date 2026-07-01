import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, GlobalAvgPool1D,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from src.preprocessing import Preprocessor
from src.utils import path_builder

path = 'data/IMDB Dataset.csv'
p = Preprocessor(path,128)
vocab_size = len(p.get_vocabulary())

model = Sequential([
    Input(shape=(128,)),
    Embedding(vocab_size, 64),
    GlobalAvgPool1D(),
    Dense(64,activation="relu"),
    Dropout(0.2),
    Dense(1,activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

x_test, x_train, x_val, y_test, y_train, y_val = p.get_train_test_val_splits()

y_train = np.where(y_train == "positive", 1.0, 0.0).astype(np.float32)
y_val   = np.where(y_val == "positive", 1.0, 0.0).astype(np.float32)
y_test  = np.where(y_test == "positive", 1.0, 0.0).astype(np.float32)

history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=64,
    validation_data= (x_val,y_val),
    shuffle=True,
    callbacks= [
    EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=2)
    ]
)

save_path = path_builder('src/models/saved_models/sentiment_classifier.keras')

model.save(save_path)

