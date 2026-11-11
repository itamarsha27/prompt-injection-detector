import argparse
from pathlib import Path
import joblib

THRESHOLD = 0.4
MODEL_PATH = Path(__file__).parent / "model.joblib"


def main():
    parser = argparse.ArgumentParser(description="Detect prompt injections")
    parser.add_argument("text", help="the prompt to check")
    args = parser.parse_args()

    pipe = joblib.load(MODEL_PATH)
    prob = pipe.predict_proba([args.text])[0, 1]
    label = "INJECTION" if prob >= THRESHOLD else "SAFE"
    print(f"{label} (probability {prob:.3f})")


if __name__ == "__main__":
    main()