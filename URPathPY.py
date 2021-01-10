import rtde_control
import rtde_receive
import math


def readPath(dataDir):
    data = []
    with open(dataDir) as f:
        for line in f:
            line = line.strip('\n')
            line = line.strip(' ')
            line = line.split('  ')
            data.append([float(x) for x in line])
    for i in range(len(data)):
        for j in range(0, 6):
            data[i][j] = data[i][j] / 180 * math.pi
    velocity = 0.5
    acceleration = 0.5
    blend_1 = 0.0
    path = []
    for i in range(len(data)):
        pathPose = []
        for j in range(6):
            pathPose.append(data[i][j])
        pathPose.append(velocity)
        pathPose.append(acceleration)
        pathPose.append(blend_1)
        path.append(pathPose)
    return path


def connectRobot(RobotIP):
    rtde_c = rtde_control.RTDEControlInterface(RobotIP)
    rtde_r = rtde_receive.RTDEReceiveInterface(RobotIP)

    return rtde_c, rtde_r

if __name__ == '__main__':
    path = readPath(r'C:\Users\小姚\Desktop\test_road.txt')
    RobotIP = "192.168.218.2"
    rtde_c, rtde_r = connectRobot(RobotIP)
    for i in range(len(path)):
        rtde_c.moveJ(path[i][0:6], 1.05, 1.4)
        print(rtde_c.getJointTorques())
    rtde_c.stopJ(0.5)
    rtde_c.stopScript()
