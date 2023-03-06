
from os import link
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class TrafficEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TrafficEnv, self).__init__()

        #network with only 1 link and 2 nodes
        # self.action_space = spaces.Box(low=np.array([10.0]), high=np.array([+10.0]), dtype=np.float32)
        # self.observation_space = spaces.Box(low=np.array([0]), high=np.array([2000]), dtype=np.float32)

        # self.num_link = 1
        
        # # self.total_veh_num = 300
        # self.state = np.array([300])
        # self.lanes_link = np.array([2])
        # self.length_link = np.array([500])
        # self.free_v_link = np.array([30])
        # self.alpha_link = np.array([0.4])
        # self.jam_density_link = np.array([2.0])
        # self.human_headway_link = np.array([3.0])



        #network with only 2 link and 2 nodes
        # self.action_space = spaces.Box(low=np.array([1.0, 1.0]), high=np.array([+10.0, +10.0]), dtype=np.float32)
        # self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([2000, 2000]), dtype=np.float32)

        # self.num_link = 2
        
        # # self.total_veh_num = 600
        # self.state = np.array([500, 100])
        # self.lanes_link = np.array([2, 2])
        # self.length_link = np.array([500, 500])
        # self.free_v_link = np.array([30, 30])
        # self.alpha_link = np.array([0.4, 0.4])
        # self.jam_density_link = np.array([2.0, 2.0])
        # self.human_headway_link = np.array([3.0, 3.0])



        #network with 3 links and one OD pair
        # self.action_space = spaces.Box(low=np.array([1.0, 1.0, 1.0]), high=np.array([+10.0, +10.0, +10.0]), dtype=np.float32)
        # self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([2000, 2000, 2000]), dtype=np.float32)

        # self.num_link = 3
        
        # # self.total_veh_num = 600
        # self.state = np.array([500, 300, 100])
        # self.lanes_link = np.array([2, 2, 2])
        # self.length_link = np.array([500, 500, 500])
        # self.free_v_link = np.array([30, 30, 30])
        # self.alpha_link = np.array([0.4, 0.4, 0.4])
        # self.jam_density_link = np.array([2.0, 2.0, 2.0])
        # self.human_headway_link = np.array([3.0, 3.0, 3.0])



        #network with 4 links and one OD pair
        self.action_space = spaces.Box(low=np.array([1.0, 1.0, 1.0, 1.0]), high=np.array([+10.0, +10.0, +10.0, +10.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([2000, 2000, 2000, 2000]), dtype=np.float32)

        self.num_link = 4
        
        # self.total_veh_num = 600
        self.state = np.array([500, 300, 300, 100])
        self.lanes_link = np.array([2, 2, 2, 2])
        self.length_link = np.array([500, 500, 500, 500])
        self.free_v_link = np.array([30, 30, 30, 30])
        self.alpha_link = np.array([0.4, 0.4, 0.4, 0.4])
        self.jam_density_link = np.array([2.0, 2.0, 2.0, 2.0])
        self.human_headway_link = np.array([3.0, 3.0, 3.0, 3.0])



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
        flow_link_in = np.zeros(self.num_link)
        flow_link_out = np.zeros(self.num_link)
        flow_on_link = np.zeros(self.num_link)
        latency_link = np.zeros(self.num_link)


        reward_cumulation = 0
        demand = np.zeros(self.num_link)
        demand[0] = 48
        # demand = self.free_v_link*cri_density_link


        for train_loop in range(100):

            #calculate input flow
            # for i in range(self.num_link):
            #     if(density_link[i]<cri_density_link[i]):
            #         flow_link_in[i] = self.free_v_link[i]*cri_density_link[i]

            #     elif(density_link[i]>self.jam_density_link[i]):
            #         flow_link_in[i] = 0

            #     else:
            #         flow_link_in[i] = self.free_v_link[i]*cri_density_link[i]*(self.jam_density_link[i]-density_link[i])/(self.jam_density_link[i]-cri_density_link[i])
                    
            #     # flow_link_in[i] = min(max(demand[i]-veh_num[i],0), flow_link_in[i])
            #     flow_link_in[i] = min(max(demand[i],0), flow_link_in[i])

            for i in range(self.num_link):
                flow_link_in[i] = max(demand[i],0)
            # veh_num = veh_num + flow_link_in
            # density_link = veh_num/self.length_link


            #calculate output flow
            for i in range(self.num_link):
                if(density_link[i]<cri_density_link[i]):
                    flow_link_out[i] = self.free_v_link[i]*density_link[i]

                elif(density_link[i]>self.jam_density_link[i]):
                    flow_link_out[i] = 0

                else:
                    flow_link_out[i] = self.free_v_link[i]*cri_density_link[i]*(self.jam_density_link[i]-density_link[i])/(self.jam_density_link[i]-cri_density_link[i])
                    
                flow_link_out[i] = min(veh_num[i], flow_link_out[i])

            veh_num = veh_num + flow_link_in - flow_link_out
            density_link = veh_num/self.length_link

            #calculate flow on link and corresponding latency
            for i in range(self.num_link):
                if(density_link[i]<cri_density_link[i]):
                    flow_on_link[i] = self.free_v_link[i]*density_link[i]
                    latency_link[i] = self.length_link[i]/self.free_v_link[i]

                elif(density_link[i]>self.jam_density_link[i]):
                    flow_on_link[i] = 0
                    # latency_link[i] = np.infty
                    latency_link[i] = 10000000000
                    # flag_done = True # failed
                    # print("Link ",str(i), "is totally jammed!")

                else:
                    flow_on_link[i] = self.free_v_link[i]*cri_density_link[i]*(self.jam_density_link[i]-density_link[i])/(self.jam_density_link[i]-cri_density_link[i])
                    latency_link[i] = self.length_link[i]*(self.jam_density_link[i]/flow_on_link[i]+(cri_density_link[i]-self.jam_density_link[i])/(self.free_v_link[i]*cri_density_link[i]))               
        

            reward_cumulation -= sum(density_link)


            #TODO: Update demand by dynamics
            
            #case of only 1 link
            # path_1_num =veh_num[0]

            # path_1_val = path_1_num*np.exp(-self.miu*(latency_link[0]))
            # total_val = path_1_val

            # _path_1 =  sum(demand)*path_1_val/(total_val)

            # demand[0] = _path_1


            #case of 2 links
            # path_1_num = veh_num[0]
            # path_2_num = veh_num[1]

            # path_1_val = path_1_num*np.exp(-self.miu*(latency_link[0]))
            # path_2_val = path_2_num*np.exp(-self.miu*(latency_link[1]))

            # total_val = path_1_val+path_2_val

            # _path_1 = sum(demand)*path_1_val/(total_val)
            # _path_2 = sum(demand)*path_2_val/(total_val)

            # demand[0] = _path_1
            # demand[1] = _path_2



            #case of 3 links
            # path_1_num = veh_num[0]
            # path_2_num = veh_num[1]
            # path_3_num = veh_num[2]

            # path_1_val = path_1_num*np.exp(-self.miu*(latency_link[0]))
            # path_2_val = path_2_num*np.exp(-self.miu*(latency_link[1]))
            # path_3_val = path_3_num*np.exp(-self.miu*(latency_link[2]))

            # total_val = path_1_val+path_2_val+path_3_val

            # _path_1 = sum(demand)*path_1_val/(total_val)
            # _path_2 = sum(demand)*path_2_val/(total_val)
            # _path_3 = sum(demand)*path_3_val/(total_val)

            # demand[0] = _path_1
            # demand[1] = _path_2
            # demand[2] = _path_3



            #case of 4 links
            path_1_num = veh_num[0]
            path_2_num = veh_num[1]
            path_3_num = veh_num[2]
            path_4_num = veh_num[3]

            path_1_val = path_1_num*np.exp(-self.miu*(latency_link[0]))
            path_2_val = path_2_num*np.exp(-self.miu*(latency_link[1]))
            path_3_val = path_3_num*np.exp(-self.miu*(latency_link[2]))
            path_4_val = path_4_num*np.exp(-self.miu*(latency_link[3]))

            total_val = path_1_val+path_2_val+path_3_val+path_4_val

            _path_1 = sum(demand)*path_1_val/(total_val)
            _path_2 = sum(demand)*path_2_val/(total_val)
            _path_3 = sum(demand)*path_3_val/(total_val)
            _path_4 = sum(demand)*path_4_val/(total_val)

            demand[0] = _path_1
            demand[1] = _path_2
            demand[2] = _path_3
            demand[3] = _path_4

            # if(train_loop<10):
            # print("Train loop:", train_loop)
            # print("Action:", action)
            # # print("total divide:", total_val)
            # print("Flow_in:", flow_link_in)
            # print("Flow_out:", flow_link_out)
            # print("Flow on link:", flow_on_link)
            # print("Latencies:", latency_link)
            # print("veh_num:",veh_num)
            # print("New demand:", demand)
            # print("Density:", density_link)
            # # print("Vals:", path_1_val, path_2_val)
            # # print("Path_Val:", np.array([path_1_val, path_2_val, path_3_val, path_4_val]))
            # print("\n")

       

        reward = reward_cumulation/100



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
        # veh_num = state_new
        # density_link = veh_num/self.length_link
        # reward = 0
        # for i in range(self.num_link):
        #     reward -= density_link[i]
        # reward = reward*self.nu

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


        
        # print("Veh num after 100 timesteps:",veh_num) 


        # if (flag_done):
        #     done = True
        # else:
        #     done = False

        info = {}
        done = False

        return self.state, reward, done, info

    def reset(self):
        # reset state to starting case

        # for 6 links occasion
        # self.state = np.array([900,200,100,100,900,200])

        # for 4 paths with same OD pair
        self.state = np.array([500, 300, 300, 100])
        return self.state
  
    def render(self, mode='human'):
        pass

    def close(self):
        pass