import random
import numpy as np
from tak.ai import TakFeatures, TakWeights
from tak.lr_data import generate_dataset
import json
from tqdm import tqdm


def train(epochs=50, lr=0.01):
    features = TakFeatures()
    keys = features.KEYS

    w = np.array([5, 40, 20, 30, 4, 5, 1, 10, 3], dtype=float)

    dataset = generate_dataset(games=300)

    for epoch in tqdm(range(epochs)):
        random.shuffle(dataset)
        loss = 0

        for feats, y in dataset:
            x = np.array(features.as_vector(feats))
            pred = np.dot(w, x)
            err = pred - y

            w -= lr * err * x
            loss += err * err

        print(f"epoch {epoch} loss {loss / len(dataset):.4f}")

    return dict(zip(keys, w))


if __name__ == "__main__":
    learned = train()

    with open("tak_weights.json", "w") as f:
        json.dump(learned, f, indent=2)
    print("Learned weights:")
    for k, v in learned.items():
        print(f"  {k}: {v:.4f}")
