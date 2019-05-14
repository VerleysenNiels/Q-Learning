# Q-Learning
Different Q-learning algorithms build with Keras with a Tensorflow back end, tested in OpenAI Gym environments.

Current algorithms:

| Algorithm                                       |CartPole   |Beamrider   |
|-------------------------------------------------|:---------:|:-----------|
|	Deep Q-Network                                  |Yes        |Yes         |
|	Double Deep Q-Network                           |Yes        |Yes         |
|	Deep Q-Network with dueling architecture        |Yes        |Yes         |
|	Double Deep Q-Network with dueling architecture |Yes        |Yes         |

## Structure
The source and results folder have the same structure, the results (csv files and extracted graphs) from the algorithms that can be found in the source folder are put in the results folder. The different algorithms are trained in the environment while running the main.py file that can be found in the same location as the algorithms. This main.py file does the necessary preprocessing and manages the results,  training and evaluation of the algorithms.

## References
Used algorithms and improvements come from following papers:
* V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, Human-level control through deep reinforcement learning, Nature, 2015
* H. van Hasselt, A. Guez and D. Silver, Deep Reinforcement Learning with Double Q-Learning, AAAI, 2016
* Z. Wang, T. Schaul, M. Hesset et al., Dueling Network Architectures for Deep Reinforcement Learning, 2015
* C. Zhang, O. Vinyals, R. Munos et al., A Study on Overfitting in Deep Reinforcement Learning, 2018

