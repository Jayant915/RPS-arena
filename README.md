# ✊🖐✌️ RPS Arena – Rock Paper Scissors AI Game

This project is a real-time Rock Paper Scissors game powered by your webcam and trained using **Teachable Machine**. It uses a machine learning model to classify hand gestures (rock, paper, scissors) and determines the winner between two players live on screen.

Built with:
- 🎥 OpenCV for webcam input
- 🤖 TensorFlow (TFSMLayer) for model inference
- 💡 Teachable Machine for training gestures

---

## 🚀 Features

- 🔍 Real-time hand gesture recognition  
- 🧠 ML model trained with Google’s Teachable Machine  
- 👨‍👩‍👧‍👦 2-player gameplay via split-screen webcam  
- 🏆 Automatic winner detection  
- ⚙️ Easy to customize or extend  

---

## 🧠 How It Works

1. The webcam is split into two zones: Player 1 and Player 2  
2. Each player shows a gesture (Rock, Paper, or Scissors)  
3. The image is preprocessed and passed to the ML model  
4. The model predicts the gesture for each player  
5. Winner is calculated using standard RPS rules and displayed on screen  


## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Jayant915/RPS-arena.git
cd RPS-arena
```

### 2. Install dependencies

```bash
pip install tensorflow opencv-python
```

### 3. Add your model

1. Train a model with 3 classes: Rock, Paper, Scissors  
2. Export the model as **TensorFlow → SavedModel format**  
3. Rename the extracted folder to `model.savedmodel` and place it in the project root  

Also create a `labels.txt` file with the following contents:

```
0 Rock
1 Paper
2 Scissors
```

### 4. Run the game

```bash
python game.py
```

---

## 📁 Project Structure

```
RPS-arena/
├── model.savedmodel/        # Trained Teachable Machine model
│   ├── saved_model.pb
│   └── variables/
├── labels.txt               # Gesture labels
├── game.py                  # Main 2-player game logic
├── check.py                 # Model loading test
└── README.md                # This file
```


## 🙌 Acknowledgments

- [TensorFlow](https://www.tensorflow.org/)
- Inspired by the classic game of Rock Paper Scissors
EOF
