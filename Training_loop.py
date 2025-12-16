from Model import SoftmaxLogisticTrainer
from prepare_data import X_train, y_train, X_val, y_val, X_test, y_test
import numpy as np

trainer = SoftmaxLogisticTrainer(
    lr=0.05,
    l2=1e-4,
    epochs=3000
)

model = trainer.fit(
    X_train, y_train,
    X_val, y_val,
    patience=20
)

np.savez(
    "softmax_logistic_params.npz",
    W=model.W,
    b=model.b,
    mean=model.mean,
    std=model.std
)
    

