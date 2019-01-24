# Navigation with DQN
Udacity Reinforcement Learning Nanodegree Project 1


##### Visualization of the agent's behaviour:

![](result.gif)


## Getting Started

Follow the instructions in the [DRLND GitHub](https://github.com/udacity/deep-reinforcement-learning#dependencies) repository to set up the Python environment. 
By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

Clone this repository in the "p1_navigation" folder of the DRLND GitHub repository.

Download the Unity environment from one of the links below. You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64
(For Windows users) Check out this [link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), download the "headless" version of the environment following this [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip).


#### Instructions
In order to train a new agent, run:
```
python train.py 
    --total_episodes ="Number of episodes to learn from" 
    --env = "Path to the downloaded Unity environment"
    --ddqn = "Whether to use the Double DQN strategy"
```
To test a trained model, run:
```
python test.py 
    --env = "Path to the downloaded Unity environment"
    --model_pth = "Path to the trained model"
```

Path to the downloaded Unity environment:

- Mac: "path/to/Banana.app"
- Windows (x86): "path/to/Banana_Windows_x86/Banana.exe"
- Windows (x86_64): "path/to/Banana_Windows_x86_64/Banana.exe"
- Linux (x86): "path/to/Banana_Linux/Banana.x86"
- Linux (x86_64): "path/to/Banana_Linux/Banana.x86_64"
- Linux (x86, headless): "path/to/Banana_Linux_NoVis/Banana.x86"
- Linux (x86_64, headless): "path/to/Banana_Linux_NoVis/Banana.x86_64"