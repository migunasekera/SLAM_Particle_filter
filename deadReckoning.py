import load_data
import glob
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse

def stringParser(fileList):
    tmp = []
    [tmp.append(fileList[i].split(".")[0]) for i in range(len(fileList))]
    return tmp

def display(FR, FL, RR, RL, sl):
    '''
    Displays the value along a specified slies

    sl: The slice of data you want to look at for all four encoding measurements
    '''
    print('FR')
    print(FR[sl])
    print('\n\nFL')
    print(FL[sl])
    print('\n\nRR')
    print(RR[sl])
    print('\n\nRL')
    print(RL[sl])

def DR(FR, FL, RR, RL, time):
    print('hello')



def newPosition(oldPosition, FR, FL, RR, RL):
    '''
    Inputs:
    oldPosition: Numpy array of (x, y, theta)
    




    Output:
    newPosition: Tuple of (x + deltaX, y + deltaY, theta + deltaTheta)
    '''
    
    distances = distCalc(FR,FL,RR,RL)
    th, R = circleCalc(distances[0],distances[1])
    #print(distances[0],distances[1])

    arcDist = R * th

    # print('Radius')
    # print(R)
    theta = oldPosition[2]
    # print('Theta')
    # print(th)

    delx = arcDist * math.cos(theta + th/2)
    dely = arcDist * math.sin(theta + th/2)
    delth = th

    delposition = np.array((delx, dely, delth))
    # delposition = np.array(delposition)

    return oldPosition + delposition

def circleCalc(right_dist, left_dist, width = 311.5):
    '''
    This is used to calculate the type of thing thats noticed.
    
    input:
        right_dist:
        left_dist
    output:
        theta:
        R:
    '''
    
    
 
    if right_dist == 0 and left_dist == 0:
        # print("They're both zero")
        return 0, 0
    
    if right_dist == left_dist: # I'm not sure how to make it a straight line right now
        # print('This should be a straight line')
        return 0.00001, 1000000
        
    
    # print('I skipped the if statement')
    theta = (right_dist - left_dist)/width
    R = (1/2) * (right_dist + left_dist) / ((right_dist - left_dist) / width)

    return theta, R

def circleDisplay(theta, R):



    # r = np.arange(0, 2, 0.01)
    # t = 2 * np.pi * r
    # print(r.shape,t.shape)

    theta = np.array(theta)
    R = np.array(R)

    print(theta.shape,R.shape)

    ax = plt.subplot(111, projection='polar')
    ax.plot(theta, R)
    # ax.set_rmax(2)
    # ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
    # ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)

    ax.set_title("A line plot on a polar axis", va='bottom')
    plt.show()

def distCalc(FR, FL, RR, RL, diameter = 254):
    '''
    input: counts/revolution, is a unit of distance. The number of counts tells how far the wheel turned in that specific time
    diameter: This is given in millimeters. Given here is the diameter of the wheel
    These are added over time. The idea is that the distance is calculated over each time interval (or perhaps, at every 5 measurements if things are going too fast?)


    
    start: Index value of beginning value
    end: Index value of end value
    diameter: Diameter of the wheel (mm)


    Output:
    FR: Front right wheel distance
    FL: Front left wheel distance
    RR: Rear right wheel distance
    RL: Rear left wheel distance

    '''
    FR_dist = 0
    FL_dist = 0
    RR_dist = 0
    RL_dist = 0

    FR_dist += FR / diameter
    FL_dist += FL / diameter
    RR_dist += RR / diameter
    RL_dist += RL / diameter

    # counts/revolutions

    return FR_dist, FL_dist, RR_dist, RL_dist




if __name__ == "__main__":
    Encoders = glob.glob('data/Encoders*')
    Test_encoders = glob.glob('2019Proj3_test/Encoders*')
    # print(Encoders)
    newEnc = stringParser(Encoders)
    testEnc = stringParser(Test_encoders)
    # print(newEnc)

    encoderDict = {'enc20': newEnc[0], 'enc21': newEnc[1], 'enc23': newEnc[2]}
    test_encoderDict = {'enc22': testEnc[0], 'enc24': testEnc[1]}
    # pos = []
    # oldPos = np.array((0,0,0)) # This is made up, but it represents the (x, y, and theta) of the robot. You will care about the present and future of this
    
    # pos.append((oldPos[0],oldPos[1]))


    # Which encoder will you choose?
    for encoder in test_encoderDict:
        print(encoder)
        FR, FL, RR, RL, time = load_data.get_encoder(test_encoderDict[encoder])
        
        size = len(FR)
        position = np.zeros((size,3)) # x, y, theta
        t = np.zeros((size,))
        r = np.zeros((size,))
        # print(t,r)
        print('Old position:')

        
    

        for i in range(1,size):
            position[i,:] = newPosition(position[i-1,:], FR[i], FL[i], RR[i], RL[i])
            # We are printing the 

            # ax.plot(oldPos[0],oldPos[1])


            # We are constantly updating the old position. We are also updating the angle, Since OldPos is a 3-tuple
            # oldPos = newPosition(oldPos,FR[i], FL[i], RR[i], RL[i])
            # here we are actually updating the odometry map
            # pos.append((oldPos[0],oldPos[1]))
            
            # But we are always recording the of theta and radius. So far, we don't really do anything with this value yet
            # t[i] = theta
            # r[i] = radius

            # pos = np.array(pos)
        x = position[:,0]; y = position[:,1]; theta = position[:,2]
        line = Line2D(x,y)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.add_line(line)
        ax.set_xlim(min(x),max(x))
        ax.set_ylim(min(y),max(y))
        # plt.plot(pos)
        plt.title(encoder)
        plt.show()

        # Total time: 123.63748502731323 seconds

        # FR, FL, RR, RL, t = load_data.get_encoder(encoderDict['enc23']) # Total time: 123.63748502731323 seconds
        
    
    
    
    # circleDisplay(t,r)

    

    # right_dist = distCalc(FR[0], FR[])
    # print(encData[4][:50])
    # encData2 = load_data.get_encoder(newEnc[2])
    #(123.63748502731323, 119.40321111679077, 94.96247100830078) #Total amount of time spent for each reading
    # print((encData[4][len(encData[4])-1] - encData[4][0]))