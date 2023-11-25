# School demo of the TicToc Game using opencv and mediapipe

This project is simple implementation of the TicToc Game using gestures.

How it works:

When the game was started, use gesture Open_Palm (👋) to move a hand, 
Closed_Fist (✊)to make a step. When game is finished just use Thumb_Up (👍)
to start a new game.


## Installation


* Clone repository:

```git clone ```
* Move to the project directory:

 ```cd TicTocGame```
* Download model:

```wget https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task```

* Install requirements:

```pip3 install -r requirements.txt```

## Run the game

To run game just run the following command:

```python3 main.py```

It should show opencv named window with game.
