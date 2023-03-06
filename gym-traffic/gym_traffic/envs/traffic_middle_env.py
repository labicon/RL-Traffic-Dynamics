from os import link
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class TrafficMidEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TrafficMidEnv, self).__init__()

        #network with 2-1-2 links setting and one OD pair


        #  (link 0) V  (link 2)
        #           |
        # O---<     |    >---D     (link 4 in the middle)
        #           v
        #  (link 1) W  (link 3)


        self.action_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]), dtype=np.float32)
        # self.action_space = spaces.MultiDiscrete([ 10, 10, 10, 10, 10 ])
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10000000000.0, 0.0, 0.0]), high=np.array([6000.0, 6000.0, 6000.0, 6000.0, 6000.0,6000.0, 6000.0, 6000.0, 6000.0, 6000.0, 0.0, 200.0, 200.0]), dtype=np.float32)
        
        #obs = [#num(auto, human), #num(auto, human), #num(auto, human), #num(auto, human), #num(auto, human), reward_, #loop_, demand_in_10]

        self.num_link = 5
        
        # self.total_veh_num = 600
        # self.state = np.array([800.0, 200.0, 400.0, 100.0, 720.0, 180.0, 480.0, 120.0, 80.0, 20.0, -100.0, 0.0, 0.0], dtype=np.float32)        #inceas the initial condition to avoid free-flow 
        self.state = np.array([1600.0, 400.0, 800.0, 200.0, 1440.0, 360.0, 960.0, 240.0, 160.0, 40.0, -100.0, 0.0, 0.0], dtype=np.float32)
        self.lanes_link = np.array([4, 2, 2, 4, 10])                                                          #give more time on the time series (horizon)
                                                                                                              #play around with learning rate
                                                                                                              # normalizing the reward (or action space)
                                                                                                              #need implement observation over training (convergence)
        self.length_link = np.array([4000, 4000, 4000, 4000, 1000])  #meters
        self.free_v_link = np.array([30, 30, 30, 30, 30])      #meters per second
        # self.alpha_link = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
        self.alpha = 0.8
        self.jam_density_link = np.array([4.0, 4.0, 4.0, 4.0, 10.0])    # num_of_veh per meter
        self.human_headway_link = np.array([3.0, 3.0, 3.0, 3.0, 3.0])



        #dynamic coef, which needed to be twitched from 1
        self.miu = 0.1     #maybe too small
        # self.nu = 1000  #new coefficient for hybrid reward function

    def step(self, action):
        flag_done = False
        


        veh_num = self.state[0:10].copy()
        density_link = np.zeros(self.num_link)
        for i in range(5):
            density_link[i] = (veh_num[2*i]+veh_num[2*i+1])/self.length_link[i]
        # density_link = veh_num/self.length_link
        action = action*10.0 + 1.0

        frac_link = np.zeros(self.num_link)
        for i in range(5):
            if(veh_num[2*i]+veh_num[2*i+1]<0.1):
                frac_link[i] = self.alpha
            else:
                frac_link[i] = veh_num[2*i]/(veh_num[2*i]+veh_num[2*i+1])

        # print("Action:", action)
        # print("State:",self.state)
        
        cri_density_link = self.lanes_link/(action*frac_link + self.human_headway_link*(1-frac_link))
        flow_link_in = np.zeros(self.num_link*2)
        flow_link_out = np.zeros(self.num_link*2)
        flow_on_link = np.zeros(self.num_link)
        latency_link = np.zeros(self.num_link)


        reward_cumulation = 0.0

        #varying demand design
        demand = np.zeros(2)
        demand[0] = self.state[12]
        # if(self.state[11]<50):
        #     demand[0] = self.state[11]*2   #from 0 to 100 in timesteps 0 to 50
        # else:
        #     demand[0] = (100.0-self.state[11])*2    #from 100 back to 0 in timesteps 50 and above
        # demand = self.free_v_link*cri_density_link



        # init flow_on_link and latency before starting the loop
        for i in range(self.num_link):
            if(density_link[i]<cri_density_link[i]):
                flow_on_link[i] = self.free_v_link[i]*density_link[i]
                latency_link[i] = self.length_link[i]/self.free_v_link[i]

            elif(density_link[i]>self.jam_density_link[i]):
                flow_on_link[i] = 0
                # latency_link[i] = np.infty
                latency_link[i] = 10000
                flag_done = True # failed
                break
                # print("Link ",str(i), "is totally jammed!")

            else:
                flow_on_link[i] = self.free_v_link[i]*cri_density_link[i]*(self.jam_density_link[i]-density_link[i])/(self.jam_density_link[i]-cri_density_link[i])
                latency_link[i] = self.length_link[i]*(self.jam_density_link[i]/flow_on_link[i]+(cri_density_link[i]-self.jam_density_link[i])/(self.free_v_link[i]*cri_density_link[i]))               
        # print("density:", density_link)
        # print("Flow_link:", flow_on_link)
        # print("Latencies:", latency_link)

        if(flag_done):
            self.state[10] = -10000000000.0
            new_reward = -10000000000.0
            done = True
            info = {}
            return self.state, new_reward, done, info


        for train_loop in range(10):

            #calculate input flow at origin
            for i in range(2):
                flow_link_in[2*i] = max(demand[i],0)*self.alpha
                flow_link_in[2*i+1] = max(demand[i],0)*(1-self.alpha)


            #calculate output flow at V and W
            flow_out_midpoint = 0
            flow_midpoints = np.zeros(4)
            for i in range(2):
                flow_out_tmp = 0.0
                if(density_link[i]<cri_density_link[i]):
                    flow_out_tmp = self.free_v_link[i]*density_link[i]

                elif(density_link[i]>self.jam_density_link[i]):
                    flow_out_tmp = 0

                else:
                    flow_out_tmp = self.free_v_link[i]*cri_density_link[i]*(self.jam_density_link[i]-density_link[i])/(self.jam_density_link[i]-cri_density_link[i])
                    
                # flow_link_out[2*i] = min(veh_num[i], flow_out_tmp)*frac_link[i]
                # flow_link_out[2*i+1] = min(veh_num[i], flow_out_tmp)*(1-frac_link[i])
                flow_link_out[2*i] = min(flow_out_tmp*frac_link[i],veh_num[2*i])
                flow_link_out[2*i+1] = min(veh_num[2*i+1], flow_out_tmp*(1-frac_link[i]))

                # #update veh_num and density on link_0 and link_1
                # veh_num[2*i] = veh_num[2*i] + flow_link_in[2*i] - flow_link_out[2*i]
                # veh_num[2*i+1] = veh_num[2*i+1] + flow_link_in[2*i+1] - flow_link_out[2*i+1]
                # density_link[i] = (veh_num[2*i]+veh_num[2*i+1])/self.length_link[i]


                flow_midpoints[2*i] = flow_link_out[2*i]
                flow_midpoints[2*i+1] = flow_link_out[2*i+1]
            

            #divide flow_midpoint[0:1] onto link_2 and link_4
            link_2_num = veh_num[4]+veh_num[5]
            link_3_num = veh_num[6]+veh_num[7]

            link_2_val = link_2_num*np.exp(-self.miu*(latency_link[2]))
            link_3_val = link_3_num*np.exp(-self.miu*(latency_link[3]+latency_link[4]))

            # if(miu*latency_link[2]<100 and miu*(latency_link[3]+latency_link[4])<100):
            #     link_2_val = link_2_num*np.exp(-miu*(latency_link[2]))
            #     link_3_val = link_3_num*np.exp(-miu*(latency_link[3]+latency_link[4]))
            # else:
            #     link_2_val = link_2_num*miu*(latency_link[3]+latency_link[4])
            #     link_3_val = link_3_num*miu*(latency_link[2])

            total_val = link_2_val+link_3_val

            # print("Need a better condition method!!!!!!!")
            if total_val==0.0:
                flow_link_in[4] = flow_midpoints[0]*link_2_num/(link_2_num+link_3_num)
                flow_link_in[5] = flow_midpoints[1]*link_2_num/(link_2_num+link_3_num)


                flow_link_in[8] = flow_midpoints[0]*link_3_num/(link_2_num+link_3_num)
                flow_link_in[9] = flow_midpoints[1]*link_3_num/(link_2_num+link_3_num)
            else:
                flow_link_in[4] = flow_midpoints[0]*link_2_val/(total_val)
                flow_link_in[5] = flow_midpoints[1]*link_2_val/(total_val)


                flow_link_in[8] = flow_midpoints[0]*link_3_val/(total_val)
                flow_link_in[9] = flow_midpoints[1]*link_3_val/(total_val)

            

            #update flow of link_4
            flow_tmp = 0.0
            if(density_link[4]<cri_density_link[4]):
                flow_tmp = self.free_v_link[4]*density_link[4]
            elif(density_link[4]>self.jam_density_link[4]):
                flow_tmp = 0
            else:
                flow_tmp = self.free_v_link[4]*cri_density_link[4]*(self.jam_density_link[4]-density_link[4])/(self.jam_density_link[4]-cri_density_link[4]) 
            # flow_link_out[8] = min(veh_num[4], flow_tmp)*frac_link[4]
            # flow_link_out[9] = min(veh_num[4], flow_tmp)*(1-frac_link[4])
            flow_link_out[8] = min(veh_num[8], flow_tmp*frac_link[4])
            flow_link_out[9] = min(veh_num[9], flow_tmp*(1-frac_link[4]))

            # veh_num[4] = veh_num[4] + flow_link_in[4] - flow_link_out[4]
            # density_link[4] = veh_num[4]/self.length_link[4]

            #flow entering link3: from link1 and link4
            flow_link_in[6] = flow_link_out[8]+flow_midpoints[2]
            flow_link_in[7] = flow_link_out[9]+flow_midpoints[3]

            #calculate output flow at destination
            flow_out_dest = np.zeros(2)
            for j in range(2):
                i = j+2
                flow_tmp = 0.0
                if(density_link[i]<cri_density_link[i]):
                    flow_tmp = self.free_v_link[i]*density_link[i]

                elif(density_link[i]>self.jam_density_link[i]):
                    flow_tmp = 0

                else:
                    flow_tmp = self.free_v_link[i]*cri_density_link[i]*(self.jam_density_link[i]-density_link[i])/(self.jam_density_link[i]-cri_density_link[i])
                    
                # flow_link_out[2*i] = min(veh_num[i], flow_tmp)*frac_link[i]
                # flow_link_out[2*i+1] = min(veh_num[i], flow_tmp)*(1-frac_link[i])
                flow_link_out[2*i] = min(veh_num[2*i], flow_tmp*frac_link[i])
                flow_link_out[2*i+1] = min(veh_num[2*i], flow_tmp*(1-frac_link[i]))

                # #update veh_num and density on link_2 and link_3
                # veh_num[i] = veh_num[i] + flow_link_in[i] - flow_link_out[i]
                # density_link[i] = veh_num[i]/self.length_link[i]

                flow_out_dest[0] += flow_link_out[2*i]
                flow_out_dest[1] += flow_link_out[2*i+1]

            #update all veh_num, density, and cre_density
            for link_idx in range(5):
                veh_num[2*link_idx] = veh_num[2*link_idx] + flow_link_in[2*link_idx] - flow_link_out[2*link_idx]
                veh_num[2*link_idx+1] = veh_num[2*link_idx+1] + flow_link_in[2*link_idx+1] - flow_link_out[2*link_idx+1]
                density_link[link_idx] = (veh_num[2*link_idx]+veh_num[2*link_idx+1])/self.length_link[link_idx]
                

                if(veh_num[2*i]+veh_num[2*i+1]<0.1):
                    frac_link[i] = self.alpha
                else:
                    frac_link[i] = veh_num[2*i]/(veh_num[2*i]+veh_num[2*i+1])
                # frac_link[i] = veh_num[2*i]/(veh_num[2*i]+veh_num[2*i+1])
            cri_density_link = self.lanes_link/(action*frac_link + self.human_headway_link*(1-frac_link))


            #update flow on link and corresponding latency
            for i in range(self.num_link):
                if(density_link[i]<cri_density_link[i]):
                    flow_on_link[i] = self.free_v_link[i]*density_link[i]
                    latency_link[i] = self.length_link[i]/self.free_v_link[i]

                elif(density_link[i]>self.jam_density_link[i]):
                    flow_on_link[i] = 0
                    # latency_link[i] = np.infty
                    latency_link[i] = 10000
                    flag_done = True # failed
                    break
                    # print("Link ",str(i), "is totally jammed!")

                else:
                    flow_on_link[i] = self.free_v_link[i]*cri_density_link[i]*(self.jam_density_link[i]-density_link[i])/(self.jam_density_link[i]-cri_density_link[i])
                    latency_link[i] = self.length_link[i]*(self.jam_density_link[i]/flow_on_link[i]+(cri_density_link[i]-self.jam_density_link[i])/(self.free_v_link[i]*cri_density_link[i]))               
        

            reward_cumulation -= sum(density_link)
            # veh_num_cal = np.zeros(5)
            # for i in range(5):
            #   veh_num_cal[i] = veh_num[2*i]+veh_num[2*i+1]
            # reward_cumulation -= sum(veh_num_cal*latency_link)/1000.0
            # reward_cumulation -= sum(latency_link)

            if (flag_done):
              break

            #TODO: Update demand for next timestep at origin by choice dynamics

            path_1_num = veh_num[0]+veh_num[1]
            path_2_num = veh_num[2]+veh_num[3]


            latency_next_node = min(latency_link[2], latency_link[3]+latency_link[4])

            path_1_val = path_1_num*np.exp(-self.miu*(latency_link[0]+latency_next_node))
            path_2_val = path_2_num*np.exp(-self.miu*(latency_link[1]+latency_link[3]))

            # if(miu*(latency_link[0]+latency_next_node)<100 and miu*(latency_link[1]+latency_link[3])<100):
            #     path_1_val = path_1_num*np.exp(-miu*(latency_link[0]+latency_next_node))
            #     path_2_val = path_2_num*np.exp(-miu*(latency_link[1]+latency_link[3]))
            # else:
            #     path_1_val = path_1_num*miu*(latency_link[1]+latency_link[3])
            #     path_2_val = path_2_num*miu*(latency_link[0]+latency_next_node)

            total_val = path_1_val+path_2_val

            if total_val==0.0:
                _path_1 = sum(demand)*path_1_num/(path_1_num+path_2_num)
                _path_2 = sum(demand)*path_2_num/(path_1_num+path_2_num)
            else:
                _path_1 = sum(demand)*path_1_val/(total_val)
                _path_2 = sum(demand)*path_2_val/(total_val)

            demand[0] = _path_1
            demand[1] = _path_2


            # # if(train_loop<10):
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
            # print("Path_Val:", np.array([path_1_val, path_2_val]))
            # print("Flow out of network:", flow_out_dest)
            # print("Cri_density:", cri_density_link)
            # print("Frac_link:", frac_link)
            # print("\n")

       


        self.state[0:2*self.num_link] = veh_num[0:2*self.num_link]
        if(self.state[11]==0):
            new_reward = reward_cumulation
        else:
            new_reward = self.state[10]+reward_cumulation
        self.state[10] = new_reward

        if(flag_done):
            new_reward = -10000000000.0
            self.state[10] = new_reward

        self.state[11] += 10

        if(self.state[11]<50):
            self.state[12] = self.state[11]*5   #from 0 to 100 in timesteps 0 to 50
        elif(self.state[11]<100):
            self.state[12] = (100.0-self.state[11])*5    #from 100 back to 0 in timesteps 50 and above
        else:
            self.state[12] = 0


        info = {}
        if(self.state[11]==200 or flag_done==True):
            done = True
        else:
            done = False
        
        # print("Veh num after 10 more timesteps:",veh_num) 
        # print("Real action:", action)
        # print("Newstate:",self.state)
        # print("New_reward:", new_reward)
        # print("Done?:",done, "\n")


        return self.state, new_reward, done, info

    def reset(self):
        # reset state to starting case

        # for 2-1-2 links setting with one OD pair
        # self.state = np.array([800.0, 200.0, 400.0, 100.0, 720.0, 180.0, 480.0, 120.0, 80.0, 20.0, -100.0, 0.0, 0.0], dtype=np.float32)
        self.state = np.array([1600.0, 400.0, 800.0, 200.0, 1440.0, 360.0, 960.0, 240.0, 160.0, 40.0, -100.0, 0.0, 0.0], dtype=np.float32)
        return self.state
  
    def render(self, mode='human'):
        pass

    def close(self):
        pass