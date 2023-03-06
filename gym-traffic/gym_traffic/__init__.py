from gym.envs.registration import register 
register(id='traffic-v0',entry_point='gym_traffic.envs:TrafficEnv',) 
register(id='traffic-v1', entry_point='gym_traffic.envs:TrafficMidEnv',)
# register(id='basic-v2',entry_point='gym_basic.envs:BasicEnv2',)