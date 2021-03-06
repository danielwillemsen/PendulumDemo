# PendulumDemo
Model-Based RL Demo for Pendulum-v0.

This demo is, very roughly inspired by the following paper from Janner et al.: When to Trust Your Model: Model-Based Policy Optimization (https://arxiv.org/abs/1906.08253).


Known requirements:
openAI gym (pip install gym), 
torch (pip install torch)

To run:
python3 demo.py

This will save videos of each episode in the "logging" folder.

Additional information:
The default settings uses 200 gradient steps every 51 time steps and uses an ensemble of 25 models. 
On my laptop, this results in every episode taking approximately 10 minutes of computing. To speed up training, computing wise, these numbers can be reduced. Might be at the cost of sample efficiency, but the parameters have not been thoroughly tested at all.

Disclaimer: this code is just a small demonstration for now. There might very well be mistakes in the code. In addition, documentation is near non-existent. Questions and or contributions are always welcome though! In addition, after a few additional runs, I have noticed that the performance is rather inconsistent, sometimes learning to balance in around ~3 episodes, and sometimes not learning anything even for 5 episodes.
