from utils.make_env import make_env
import numpy as np
import time

env_id = 'multi_speaker_listener'  #8+8  (10)
env = make_env(env_id, discrete_action=True)
env.seed(1000)
np.random.seed(1000)

obs = env.reset()
for _ in range(1000):
    I = np.eye(5)
    actions = []
    for _ in range(20):
        action = np.random.choice(list(range(5)))
        print(I[action])
        actions.append(I[action])
    next_obs, rewards, dones, infos = env.step(actions)
    env._render(close=False)  # close=False
    obs = next_obs
    time.sleep(0.1)


