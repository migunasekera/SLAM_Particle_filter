# Adapted only to plot out direction of odometry data
# load data from wheel encoders and lidar (just wheel encoders for now)
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import argparse


class Robot:
    def __init__(self):
        # Physical data on robot (meters)
        self.wheel_diameter =.254 
        self.axle_width = .47265
        # Wheel encoder data
        self.wheel_encoder_FR = None
        self.wheel_encoder_FL = None
        self.wheel_encoder_RR = None
        self.wheel_encoder_RL = None
        self.wheel_encoder_ts = None
        self.last_we_reading = None
        # Trajectory data
        self.current_pose = np.array([0.0, 0.0, 0.0])
        self.prev_delta = np.array([0.0, 0.0])
        self.delta = np.array([0.0,0.0])

    
    def get_encoder(self, file_name):
      '''
      Taken from sample code (ie test_load_data)
      '''

      data = io.loadmat(file_name+".mat")        

      self.wheel_encoder_FR = np.double(data['Encoders']['counts'][0,0][0])
      self.wheel_encoder_FL = np.double(data['Encoders']['counts'][0,0][1])
      self.wheel_encoder_RR = np.double(data['Encoders']['counts'][0,0][2])
      self.wheel_encoder_RL = np.double(data['Encoders']['counts'][0,0][3])
      self.wheel_encoder_ts = np.double(data['Encoders']['ts'][0,0][0])
      self.last_we_reading = -1

        
    def deadReckoning(self, ts):
        '''
        Get dead reckoning of wheel odometry at time stamp ts
        
        input: ts
        '''
        
        tick_length = (1.0 / 360.0) * np.pi * self.wheel_diameter
        x_prev = self.current_pose[0]
        y_prev = self.current_pose[1]
        theta_prev = self.current_pose[2]
        
#         print(np.absolute(self.wheel_encoder_ts - ts)[1000:1500])
        closest_we_reading = np.argmin(np.absolute(self.wheel_encoder_ts - ts))
        if closest_we_reading == self.wheel_encoder_ts.shape[0] - 1:
          closest_we_reading -= 1
        if closest_we_reading < self.last_we_reading:
          print("ERROR: Provided time step is less than previous...")
          sys.exit(1)
        
        
        # Sum of readings over a period of time encoders spit out the number of readings PER time step
        rl = np.sum(self.wheel_encoder_RL[closest_we_reading : ts + 1])
        rr = np.sum(self.wheel_encoder_RL[closest_we_reading : ts + 1])
        
        sl = rl * tick_length
        sr = rr * tick_length
        
        s = (sl + sr) / 2
        
        self.prev_delta[0] = s
        self.prev_delta[1] = s / (1.5 * self.axle_width)
        
        
        
        self.current_pose[2] = theta_prev + self.prev_delta[1]
        
        self.delta[0] = s * np.cos(self.current_pose[2])
        self.delta[1] = s * np.sin(self.current_pose[2])
        
        self.current_pose[0] = x_prev + s * np.cos(self.current_pose[2])
        self.current_pose[1] = y_prev + s * np.sin(self.current_pose[2])
        
        # reset the last wheel encoder reading for the next iteration
#         print(self.last_we_reading)
#         print(closest_we_reading)
        
        self.last_we_reading = closest_we_reading
        
        return self.current_pose, self.delta
    
        # Initialize the arrays for the occupancy grid 
    def initialize_map(self):
        # Adapted from MapUtils test
        self.map = {}
        self.map['res']   = 0.05 #meters
        self.map['xmin']  = -30  #meters
        self.map['ymin']  = -30
        self.map['xmax']  =  30
        self.map['ymax']  =  30 
        self.map['sizex']  = int(np.ceil((self.map['xmax'] - self.map['xmin']) / self.map['res'] + 1))
        self.map['sizey']  = int(np.ceil((self.map['ymax'] - self.map['ymin']) / self.map['res'] + 1))
        self.map['map'] = np.zeros((self.map['sizex'],self.map['sizey'])) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("encoderPath", help = "Provide the path for encoder", type = str)
    parser.add_argument("--save", help = "Save the result shown", action = "store_true")

    args = parser.parse_args()

    path = args.encoderPath
    path = path[:-4]
    
    robot = Robot()
    robot.get_encoder(path)
    robot.initialize_map()

    # init
    dist = np.zeros((len(robot.wheel_encoder_ts),len(robot.current_pose)))
    delta = np.zeros((len(robot.wheel_encoder_ts),len(robot.delta)))

    # Dead reckoning, to save positions
    for i in np.arange(1, len(robot.wheel_encoder_ts), 3):
        dist[i,...], delta[i,...] = robot.deadReckoning(i)
    
    # Easier writing
    # tmp = dist[:-1000]

    X = dist[:,0]
    Y = dist[:,1]
    
    X_grad = delta[:,0]
    Y_grad = delta[:,1]

    # Discretize pose positions for plotting on map
    x = robot.map['sizex']
    y = robot.map['sizey']
    xbins = np.arange(-x //2, x // 2)
    ybins = np.arange(-y //2, y // 2)
    print(xbins)
    print(len(ybins))
    print(len(dist))

    # print(ybins[-10:])
    
    
    # X = np.digitize(dist[:,0], xbins, right=False)
    # print(np.min(X))

    
    # # X_grad = np.digitize(Dtmp[:,0], xbins)
    # Y = np.digitize(dist[:,1], ybins, right=False)
    # plt.hist(X)
    # plt.title("X-axis distribution")
    # plt.show()
    
    # plt.hist(Y)
    # plt.title("Y-axis distribution")
    # plt.show()

    
    # Y_grad = np.digitize(Dtmp[:,0], ybins)

    # pose = np.vstack((X,Y)).T
    # print(pose.shape)

    # Plotting

    # plt.scatter(X,Y)
    # plt.scatter(X[:2000], Y[:2000], c='red')
    # plt.scatter(X[2000:], Y[2000:],c='blue')
  
    
    plt.quiver(X,Y, X_grad, Y_grad)
    plt.title(f'Odometry direction of {path[5:]}')
    
    if args.save:
      print("vector plot saved")
      plt.savefig(f'Results/{path[5:]}_2.png')
    plt.show()