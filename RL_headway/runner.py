# from gym_traffic.envs.traffic_basic_env import TrafficEnv
# from gym_traffic.envs.traffic_middle_env import TrafficMidEnv
import numpy as np
import gym
# from models.policy import random_policy
# from options import parse_options
import logging as log

from stable_baselines3 import PPO, A2C, DDPG, DQN
from stable_baselines3.common.env_checker import check_env

import wandb
from wandb.integration.sb3 import WandbCallback


config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 4000000,
    "env_name": "Traffic-dynamics-braess",
    "Model": "PPO",
    "learning-rate": 0.0002,
    "miu": 0.1,
    "alpha": 0.8,
}
run = wandb.init(
    project="Traffic-dynamics-ICRAreview-braess",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)


# Set logger display format
log.basicConfig(format='[%(asctime)s] [INFO] %(message)s', 
                datefmt='%d/%m %H:%M:%S',
                level=log.INFO)

if __name__ == "__main__":
    """Main program."""

    # args, args_str = parse_options()
    # env = gym.make(args.environment_name)
    # env = TrafficEnv()
    env = TrafficMidEnv()
    obs = env.reset()

    # check_env(env)
    # print("Checked!")
   
    # action = random_policy(obs['observation'], obs['desired_goal'],env)
    # obs, reward, done, info = env.step(action)
    # model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.1)
    # model = DDPG("MlpPolicy", env, verbose=1, learning_rate=0.1)

    # action, _states = model.predict(obs, deterministic=True)

    # obs, reward, done, info = env.step(action)
    # # env.render()
    # print(env.state)
    # if done:
    #     obs = env.reset()



    model = PPO(config["policy_type"], env, verbose=1, learning_rate=config["learning-rate"], tensorboard_log=f"runs/{run.id}")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    
    print("Trained!")
    run.finish()

    
    obs = env.reset()
    for i in range(2):
        print("New instance started! \n")

        # if i%10==0:
        #     print("Road status before:", obs)
        #     print("Action:", action)
        done = False
        while done==False:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            # env.render() 
            # print("Action:", action)
            print("Headways(autonomous vehicles):", action*10.0+1.0)
            # print("Next state:", _states)
            print("Obs:", obs)
            # print("Road status new:", obs)
            print("Reward:", reward)
            print("Done?:", done)
            print("\n")
            # if i%10==0:
            #     print("Road status new:", obs)
            #     print("Reward:", reward)
        obs = env.reset()
    env.close()


    # If we want, we can substitute a goal here and re-compute
    # the reward. For instance, we can just pretend that the desired
    # goal was what we achieved all along.
    # substitute_goal = obs['achieved_goal'].copy()
    # substitute_reward = env.compute_reward( obs['achieved_goal'], substitute_goal, info)
    # print('reward is {}, substitute_reward is {}'.format(reward, substitute_reward))