# Q-Learning
Different Q-learning algorithms build with Keras with a Tensorflow back end, tested in OpenAI Gym environments.

Current algorithms:

| Algorithm                                       |CartPole   |Atari   |
|-------------------------------------------------|:---------:|:-------|
|	Deep Q-Network                                  |Yes        |Yes     |
|	Double Deep Q-Network                           |Yes        |Yes     |
|	Deep Q-Network with dueling architecture        |Yes        |No      |
|	Double Deep Q-Network with dueling architecture |Yes        |No      |

Beamrider-v4 was used as Atari environment, but with some small changes this can be used in other atari environments as well.

## References
Used algorithms come from following papers:
* V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, Human-level control through deep reinforcement learning, Nature, 2015
* H. van Hasselt, A. Guez and D. Silver, Deep Reinforcement Learning with Double Q-Learning, AAAI, 2016
* Z. Wang, T. Schaul, M. Hesset et al., Dueling Network Architectures for Deep Reinforcement Learning, 2015
