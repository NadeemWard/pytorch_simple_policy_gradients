# pytorch_simple_policy_gradients
Reimplementation of simple policy gradient algorithms such as REINFORCE and one-step Actor-Critic methods with and without a baseline.

An example of how to run reinforce:

```bash
> python main.py --namestr="name of experiment" --env-name <Name_of_{gym/mujoco}_env> --baseline {True/False} --num-episodes 4000
```
