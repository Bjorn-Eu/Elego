Elego is program that plays the board game Go, and in particular what is known as Atari Go or Capture Go, a simplified version of go often used as an introduction for beginners. The program is written as an initial project to get used to Python and deep learning using the PyTorch framework. 

Elego is constructed similar to other modern go programs, combining MCTS together with a neural network outputting a value and a policy of potential moves for each board states. Our main reason for working with Atari Go as opposed to the full game is that it makes game length much shorter which enable us to train it to decent strength even with our limited hardware (a prehistoric laptop without a gpu).

If you want to play capture go against Elego, using a pretrained network, we have for now added a simple user interface written in PyGame. This can be run by running the "playgui.py" file and requires that PyTorch and Pygame are both available. 
Pygame can be installed by running:
```
python3 -m pip install -U pygame --user
```
![Elego](https://github.com/Bjorn-Eu/Elego/blob/main/Elego.png)

Some of the resources we found useful while working on this program:
- The book "Deep learning and the game of go" by Pumperla and Ferguson, which provides an accessible introduction to writing an AI for Go.
- The 'minigo' project, a go AI written in Python. 
- The 'KataGo' project, the currently strongest available open source go AI.

