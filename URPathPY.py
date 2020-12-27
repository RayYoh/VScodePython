import rtde_control
import rtde_receive
import  math

dataDir = r'C:\Users\小姚\Desktop\test_road.txt'
data = []
with open(dataDir) as f:
    for line in f:
        line = line.strip('\n')
        line = line.strip(' ')
        line = line.split('  ')
        data.append([float(x) for x in line])
for i in range(len(data)):
    for j in range(0,4):
        data[i][j] = data[i][j] / 100
    for j in range(3,6):
        data[i][j] = data[i][j] / 180 * math.pi


rtde_c = rtde_control.RTDEControlInterface("192.168.218.2")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.218.2")
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
print(path)
path_pose1 = [-0.143, -0.435, 0.20, -0.001, 3.12, 0.04, velocity, acceleration, blend_1]
path_pose2 = [-0.143, -0.51, 0.21, -0.001, 3.12, 0.04, velocity, acceleration, blend_1]
path_pose3 = [-0.32, -0.61, 0.31, -0.001, 3.12, 0.04, velocity, acceleration, blend_1]
path = [path_pose1, path_pose2, path_pose3]
print(path)
# Send a linear path with blending in between - (currently uses separate script)
rtde_c.moveL(path)
rtde_c.stopScript()
print(rtde_r.getActualTCPPose())
