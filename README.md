# DRDQL-4-Atari
Pytorch implementation of Deep Recurrent (Double) Q-Learning. An overview is available and can be seen in the [`AtariDRDQN`](AtariDRDQN.pptx) powerpoint.

## Comparison
Here follows a comparison between the different method tested, showing that the DRDQN architecture gives the best results.

### Space Invaders

|    | DQN  | DDQN | DRQN | DRDQN |
| -- | --- |:-----:|:----:|:-----:|
| GIF | ![alt text](https://github.com/LucaCorvitto/DRDQL-4-Atari/blob/main/gifs/DQN_SI.gif)| ![alt text](https://github.com/LucaCorvitto/DRDQL-4-Atari/blob/main/gifs/DDQN_SI.gif) | ![alt text](https://github.com/LucaCorvitto/DRDQL-4-Atari/blob/main/gifs/DRQN_SI.gif) | ![alt text](https://github.com/LucaCorvitto/DRDQL-4-Atari/blob/main/gifs/DRDQN_SI.gif) |
| Mean Value | 82 | 135 | 298 | 358 |-->

### Q*Bert
|    | DQN  | DDQN | DRQN | DRDQN |
| -- | ---- |:----:|:----:|:-----:|
| GIF | ![alt text](https://github.com/LucaCorvitto/DRDQL-4-Atari/blob/main/gifs/DQN_QB.gif) | ![alt text](https://github.com/LucaCorvitto/DRDQL-4-Atari/blob/main/gifs/DDQN_QB.gif) | ![alt text](https://github.com/LucaCorvitto/DRDQL-4-Atari/blob/main/gifs/DRQN_QB.gif) | ![alt text](https://github.com/LucaCorvitto/DRDQL-4-Atari/blob/main/gifs/DRDQN_QB.gif) |
| Mean Value | 121 | 285 | 253 | 312 |


## References
The baseline codes for DQN and DDQN are taken from [Hauf3n](https://github.com/Hauf3n) in its repository [DDQN-Atari-PyTorch](https://github.com/Hauf3n/DDQN-Atari-PyTorch) and slighlty modified.

The architectures of DRQN and DRDQN are taken from the paper [Performing Deep Recurrent Double Q-Learning for Atari Games](https://www.researchgate.net/publication/340060033_Performing_Deep_Recurrent_Double_Q-Learning_for_Atari_Games)
