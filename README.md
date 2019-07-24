# pytorch_simple_policy_gradients
Reimplementation of simple policy gradient algorithms such as REINFORCE and one-step Actor-Critic methods with and without a baseline.

An example of how to run reinforce:

```bash
> python main_reinforce.py --namestr="CartPole Reinforce Baseline" --env-name CartPole-v0 --baseline True --action-space discrete --num-episodes 2000

```
