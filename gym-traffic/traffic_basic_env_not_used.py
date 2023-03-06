from os import link
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class TrafficEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TrafficEnv, self).__init__()
        # # action_space: auto_headway on each link
        # self.action_space = spaces.Box(low=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), high=np.array([+10.0, +10.0, +10.0, +10.0, +10.0, +10.0]), dtype=np.float32)
        # # obs_space: number of vehicle on each link
        # # self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0]), high=np.array([1000, 1200, 600, 600, 960, 840]), dtype=np.float32)
        # self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0]), high=np.array([2000, 2000, 2000, 2000, 2000, 2000]), dtype=np.float32)
        
        # self.num_link = 6
        
        # self.total_veh_num = 1100
        # self.state = np.array([900,200,100,100,900,200])
        # self.lanes_link = np.array([4, 4, 2, 2, 2, 2])
        # self.length_link = np.array([500, 600, 500, 500, 800, 700])
        # self.free_v_link = np.array([40, 50, 40, 40, 20, 40])
        # self.alpha_link = np.array([0.4, 0.4, 0.2, 0.2, 0.2, 0.4])
        # self.jam_density_link = np.array([2.0, 2.0, 1.2, 1.2, 1.2, 1.2])
        # self.human_headway_link = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])

        # #simplest network with 4 links and 2 nodes (1 OD pair)
        # self.action_space = spaces.Box(low=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), high=np.array([+10.0, +10.0, +10.0, +10.0, +10.0, +10.0, +10.0, +10.0]), dtype=np.float32)
        # self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0]), high=np.array([2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]), dtype=np.float32)

        # self.num_link = 8
        
        # self.total_veh_num = 1020
        # self.state = np.array([800,200,10,10,800,200,10,10])
        # self.lanes_link = np.array([2, 2, 2, 2, 2, 2, 2, 2])
        # self.length_link = np.array([500, 600, 500, 500, 500, 600, 500, 500])
        # self.free_v_link = np.array([50, 50, 50, 50, 50, 50, 50, 50])
        # self.alpha_link = np.array([0.2, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.4])
        # self.jam_density_link = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        # self.human_headway_link = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])

        #network with only 1 link and 2 nodes
        self.action_space = spaces.Box(low=np.array([10.0]), high=np.array([+10.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([2000]), dtype=np.float32)

        self.num_link = 1
        
        self.total_veh_num = 500
        self.state = np.array([500])
        self.lanes_link = np.array([2])
        self.length_link = np.array([500])
        self.free_v_link = np.array([50])
        self.alpha_link = np.array([0.2])
        self.jam_density_link = np.array([2.0])
        self.human_headway_link = np.array([3.0])

        #network with only 2 link and 2 nodes
        # self.action_space = spaces.Box(low=np.array([1.0, 1.0]), high=np.array([+10.0, +10.0]), dtype=np.float32)
        # self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([2000, 2000]), dtype=np.float32)

        # self.num_link = 2
        
        # self.total_veh_num = 520
        # self.state = np.array([500, 20])
        # self.lanes_link = np.array([2, 2])
        # self.length_link = np.array([500, 600])
        # self.free_v_link = np.array([50, 60])
        # self.alpha_link = np.array([0.2, 0.2])
        # self.jam_density_link = np.array([2.0, 2.0])
        # self.human_headway_link = np.array([3.0, 3.0])



        #dynamic coef, which needed to be twitched from 1
        self.miu = 0.05
        self.nu = 1000  #new coefficient for hybrid reward function

    def step(self, action):
        flag_done = False
        # print("Action:", action)
        # print("State:",self.state)


        veh_num = self.state.copy()
        density_link = veh_num/self.length_link
        
        cri_density_link = self.lanes_link/(action*self.alpha_link + self.human_headway_link*(1-self.alpha_link))
        flow_link = np.zeros(self.num_link)
        latency_link = np.zeros(self.num_link)

        # old definition of flow and latency
        # for i in range(self.num_link):
        #     if(density_link[i]<cri_density_link[i]):
        #         flow_link[i] = self.free_v_link[i]*density_link[i]
        #         latency_link[i] = self.length_link[i]/self.free_v_link[i]

        #     elif(density_link[i]>self.jam_density_link[i]):
        #         flow_link[i] = 0
        #         # latency_link[i] = np.infty
        #         latency_link[i] = 1000000
        #         flag_done = True # failed
        #         # print("Link ",str(i), "is totally jammed!")

        #     else:
        #         flow_link[i] = self.free_v_link[i]*cri_density_link[i]*(self.jam_density_link[i]-density_link[i])/(self.jam_density_link[i]-cri_density_link[i])
        #         latency_link[i] = self.length_link[i]*(self.jam_density_link[i]/flow_link[i]+(cri_density_link[i]-self.jam_density_link[i])/(self.free_v_link[i]*cri_density_link[i]))
        # reward = 0
        # for i in range(self.num_link):
        #     reward -= self.state[i]*latency_link[i]
        # print("reward:", reward)

        # new definition of flow and latency
        # for i in range(self.num_link):
        #     if(density_link[i]<cri_density_link[i]):
        #         flow_link[i] = self.free_v_link[i]*density_link[i]

        #     elif(density_link[i]>self.jam_density_link[i]):
        #         flow_link[i] = 0

        #     else:
        #         flow_link[i] = self.free_v_link[i]*cri_density_link[i]*(self.jam_density_link[i]-density_link[i])/(self.jam_density_link[i]-cri_density_link[i])
            
        #     latency_link[i] = density_link[i]
        # reward = 0
        # for i in range(self.num_link):
        #     reward -= density_link[i]
        # reward = reward*100
        # print("reward:", reward)

        # hybrid definition, updated Feb. 12th
        for i in range(self.num_link):
            if(density_link[i]<cri_density_link[i]):
                flow_link[i] = self.free_v_link[i]*density_link[i]
                latency_link[i] = self.length_link[i]/self.free_v_link[i]

            elif(density_link[i]>self.jam_density_link[i]):
                flow_link[i] = 0
                # latency_link[i] = np.infty
                latency_link[i] = 10000000000
                flag_done = True # failed
                # print("Link ",str(i), "is totally jammed!")

            else:
                flow_link[i] = self.free_v_link[i]*cri_density_link[i]*(self.jam_density_link[i]-density_link[i])/(self.jam_density_link[i]-cri_density_link[i])
                latency_link[i] = self.length_link[i]*(self.jam_density_link[i]/flow_link[i]+(cri_density_link[i]-self.jam_density_link[i])/(self.free_v_link[i]*cri_density_link[i]))               
        # reward = 0
        # for i in range(self.num_link):
        #     reward -= self.state[i]*latency_link[i] + density_link[i]*self.nu         




        # calculate vehcle number on each link for next time step
        # need future modification for generalizing with different path number!
        

        # for this simple case, we have four paths: {0,4},{0,2,5},{1,5},{1,3,4}
        # if(flag_done==False):
        #     state_new = np.zeros(self.num_link, dtype=np.float32)
        #     path_1_num = self.state[0]-self.state[2]
        #     path_2_num = self.state[2]
        #     path_3_num = self.state[1]-self.state[3]
        #     path_4_num = self.state[3]

        #     path_1_val = path_1_num*np.exp(-self.miu*(latency_link[0]+latency_link[4]))
        #     path_2_val = path_2_num*np.exp(-self.miu*(latency_link[0]+latency_link[2]+latency_link[5]))
        #     path_3_val = path_3_num*np.exp(-self.miu*(latency_link[1]+latency_link[5]))
        #     path_4_val = path_4_num*np.exp(-self.miu*(latency_link[1]+latency_link[3]+latency_link[4]))
        #     total_val = path_1_val+path_2_val+path_3_val+path_4_val
        #     # print(self.state)
        #     # print("total divide:", total_val)
        #     # print("Latencies:", latency_link)
        #     # print("Path_Val:", np.array([path_1_val, path_2_val, path_3_val, path_4_val]))

        #     _path_1 =  self.total_veh_num*path_1_val/(total_val)
        #     _path_2 =  self.total_veh_num*path_2_val/(total_val)
        #     _path_3 =  self.total_veh_num*path_3_val/(total_val)
        #     _path_4 =  self.total_veh_num*path_4_val/(total_val)
        #     state_new[0] = _path_1+_path_2
        #     state_new[1] = _path_3+_path_4
        #     state_new[2] = _path_2
        #     state_new[3] = _path_4
        #     state_new[4] = _path_1+_path_4
        #     state_new[5] = _path_2+_path_3

        #     if(np.sum(np.abs(self.state-state_new))<10):
        #         flag_done = True
            
        #     self.state = state_new


        # below is the dynamics of case with 4 paths of same OD pair
        # if(flag_done==False):
        #     state_new = np.zeros(self.num_link, dtype=np.float32)
        #     path_1_num = self.state[0]
        #     path_2_num = self.state[1]
        #     path_3_num = self.state[2]
        #     path_4_num = self.state[3]

        #     path_1_val = path_1_num*np.exp(-self.miu*(latency_link[0]+latency_link[4]))
        #     path_2_val = path_2_num*np.exp(-self.miu*(latency_link[1]+latency_link[5]))
        #     path_3_val = path_3_num*np.exp(-self.miu*(latency_link[2]+latency_link[6]))
        #     path_4_val = path_4_num*np.exp(-self.miu*(latency_link[3]+latency_link[7]))
        #     total_val = path_1_val+path_2_val+path_3_val+path_4_val

        #     # print("State:",self.state)
        #     # print("Action:", action)
        #     # print("total divide:", total_val)
        #     # print("Latencies:", latency_link)
        #     # print("Path_Val:", np.array([path_1_val, path_2_val, path_3_val, path_4_val]))
        #     # print("\n")

        #     _path_1 =  self.total_veh_num*path_1_val/(total_val)
        #     _path_2 =  self.total_veh_num*path_2_val/(total_val)
        #     _path_3 =  self.total_veh_num*path_3_val/(total_val)
        #     _path_4 =  self.total_veh_num*path_4_val/(total_val)
        #     state_new[0] = _path_1
        #     state_new[1] = _path_2
        #     state_new[2] = _path_3
        #     state_new[3] = _path_4
        #     state_new[4] = _path_1
        #     state_new[5] = _path_2
        #     state_new[6] = _path_3
        #     state_new[7] = _path_4


        #     if(np.sum(np.abs(self.state-state_new))<10):
        #         flag_done = True
            
        #     self.state = state_new

        #simplest case: one path and 2 nodes
        if(flag_done==False):
            state_new = np.zeros(self.num_link, dtype=np.float32)
            path_1_num = self.state[0]

            path_1_val = path_1_num*np.exp(-self.miu*(latency_link[0]))
            total_val = path_1_val

            # print("State:",self.state)
            # print("Action:", action)
            # print("total divide:", total_val)
            print("Latencies:", latency_link)
            # print("Path_Val:", np.array([path_1_val, path_2_val, path_3_val, path_4_val]))
            # print("\n")

            _path_1 =  self.total_veh_num*path_1_val/(total_val)

            state_new[0] = _path_1



            if(np.sum(np.abs(self.state-state_new))<10):
                flag_done = True
            
            self.state = state_new
        

        #simplest case: two paths and 2 nodes
        # if(flag_done==False):
        #     state_new = np.zeros(self.num_link, dtype=np.float32)
        #     path_1_num = self.state[0]
        #     path_2_num = self.state[1]

        #     path_1_val = path_1_num*np.exp(-self.miu*(latency_link[0]))
        #     path_2_val = path_2_num*np.exp(-self.miu*(latency_link[1]))

        #     total_val = path_1_val+path_2_val

        #     # print("State:",self.state)
        #     # print("Action:", action)
        #     # print("total divide:", total_val)
        #     # print("Latencies:", latency_link)
        #     # print("Path_Val:", np.array([path_1_val, path_2_val]))
        #     # print("\n")

        #     _path_1 = self.total_veh_num*path_1_val/(total_val)
        #     _path_2 = self.total_veh_num*path_2_val/(total_val)

        #     state_new[0] = _path_1
        #     state_new[1] = _path_2



        #     if(np.sum(np.abs(self.state-state_new))<10):
        #         flag_done = True
            
        #     self.state = state_new





        #calculate the reward after one step of natural evolve
        # veh_num = state_new
        # density_link = veh_num/self.length_link
        # cri_density_link = self.lanes_link/(action*self.alpha_link + self.human_headway_link*(1-self.alpha_link))
        # flow_link = np.zeros(self.num_link)
        # latency_link = np.zeros(self.num_link)
        # for i in range(self.num_link):
        #     if(density_link[i]<cri_density_link[i]):
        #         flow_link[i] = self.free_v_link[i]*density_link[i]
        #         latency_link[i] = self.length_link[i]/self.free_v_link[i]

        #     elif(density_link[i]>self.jam_density_link[i]):
        #         flow_link[i] = 0
        #         # latency_link[i] = np.infty
        #         latency_link[i] = 10000000000
        #         flag_done = True # failed
        #         # print("Link ",str(i), "is totally jammed!")

        #     else:
        #         flow_link[i] = self.free_v_link[i]*cri_density_link[i]*(self.jam_density_link[i]-density_link[i])/(self.jam_density_link[i]-cri_density_link[i])
        #         latency_link[i] = self.length_link[i]*(self.jam_density_link[i]/flow_link[i]+(cri_density_link[i]-self.jam_density_link[i])/(self.free_v_link[i]*cri_density_link[i]))               
        # reward = 0
        # for i in range(self.num_link):
        #     reward -= self.state[i]*latency_link[i] + np.var(density_link)*self.nu


        # new reward function with density as latency
        veh_num = state_new
        density_link = veh_num/self.length_link
        reward = 0
        for i in range(self.num_link):
            reward -= density_link[i]
        reward = reward*self.nu

        # hybrid reward function, updated Feb.12th
        # reward = 0
        # for i in range(self.num_link):
        #     reward -= self.state[i]*latency_link[i] + density_link[i]*self.nu


        # hybrid2 reward function: density + var(delay of time)
        # reward = 0
        # for i in range(self.num_link):
        #     reward -= density_link[i]*self.nu
        # delay_list = np.zeros(self.num_link)
        # for i in range(self.num_link):
        #     delay_list[i] = self.state[i]*latency_link[i]
        # reward -= np.var(delay_list)


        
        # print("State:",self.state) 


        if (flag_done):
            done = True
        else:
            done = False

        info = {}

        return self.state, reward, done, info

    def reset(self):
        # reset state to starting case

        # for 6 links occasion
        # self.state = np.array([900,200,100,100,900,200])

        # for 4 paths with same OD pair
        self.state = np.array([500])
        return self.state
  
    def render(self, mode='human'):
        pass

    def close(self):
        pass