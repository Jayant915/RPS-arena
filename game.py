from keras.layers import TFSMLayer
from keras import Model, Input
import cv2
import numpy as np

# Load model from SavedModel folder
layer = TFSMLayer("model.savedmodel", call_endpoint="serving_default")

# Wrap TFSMLayer in a Keras model
inp = Input(shape=(224, 224, 3))
model = Model(inputs=inp, outputs=layer(inp))

# Load class names
with open("labels.txt", "r") as f:
    class_names = [line.strip()[2:] for line in f.readlines()]

# Normalize + predict
def predict_class(img):
    img = cv2.resize(img, (224, 224))
    img = np.asarray(img, dtype=np.float32)
    img = (img / 127.5) - 1
    img = np.expand_dims(img, axis=0)
    output = model.predict(img)
    if isinstance(output, dict):  # Teachable Machine returns dict
        output = list(output.values())[0]
    idx = np.argmax(output)
    return class_names[idx], float(output[0][idx])

# Determine winner
def get_winner(p1, p2):
    if p1 == p2:
        return "It's a Tie!"
    elif (p1 == "Rock" and p2 == "Scissors") or \
         (p1 == "Paper" and p2 == "Rock") or \
         (p1 == "Scissors" and p2 == "Paper"):
        return "Player 1 Wins!"
    else:
        return "Player 2 Wins!"

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Define player zones
    roi1 = frame[100:324, 50:274]
    roi2 = frame[100:324, 350:574]

    # Get predictions
    p1_move, p1_conf = predict_class(roi1)
    p2_move, p2_conf = predict_class(roi2)
    result = get_winner(p1_move, p2_move)

    # Draw boxes
    cv2.rectangle(frame, (50, 100), (274, 324), (255, 0, 0), 2)
    cv2.rectangle(frame, (350, 100), (574, 324), (0, 255, 0), 2)

    # Show results
    cv2.putText(frame, f"P1: {p1_move} ({int(p1_conf * 100)}%)", (50, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"P2: {p2_move} ({int(p2_conf * 100)}%)", (350, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, result, (150, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    # Display window
    cv2.imshow("Rock Paper Scissors - 2 Player", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
