# Pytorch simple policy gradients methods                                                       
Implementation of simple policy gradient algorithms such as REINFORCE and one-step Actor-Critic methods with and without a baseline. This repo supports both **continuous** and **discrete** environments in OpenAI gym.                                                       
## Requirements                                         
    - Python 3.xx                                                                                            - Pytorch                                                                                            
    - OpenAI gym                                                                                         
    - Mujoco (optional)                                                                                 
# How to run and Configuration

There are two folder for each of the two methods implemented in this repo (one-step Actor Critic and REINFORCE).  

An example of how to run reinforce:                     
```bash                                                                                                  
> python main_reinforce.py --namestr="CartPole Reinforce Baseline" --env-name CartPole-v0 --baseline True --action-space discrete --num-episodes 2000                         
```
An example of how to run the actor-crtic method:                     
```bash                                                  
> python main_actor_critic.py --namestr="Cartpole" --env-name CartPole-v0 baseline True --action-space discrete --num-episodes 5000
```

notice that there are a couple of command line arugments that can be used to customize the experiments.

    - "env-name": the name of the OpenAI gym environemt
    - "baseline": whether to use a baseline for the update or not.
    - "action-space": Whether the action space is continous or not.
    - "num_episodes": number of episode you want to run the training process for.
    
## Algorithms                                                                                          
The main difference between the two algorithms implemented are in the TD error ($\delta$) used for updating the policy (and state-value) network. 

REINFORCE uses a Monte Carlo type error that is sometimes called "rewards-to-go" (https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#don-t-let-the-past-distract-you). 
