# import numpy as np
# import torch
# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# a = [[1, -3], [4, 5]]
# x = torch.unsqueeze(torch.tensor(a[0]), dim=1)
# x_temp = torch.unsqueeze(torch.tensor(a[1]), dim=1)
# x = torch.cat((x, x_temp), 1)
# x= x.to(device)
# x_abs=map(abs,x.data.cpu().numpy())
# x_abs_ls=list(x_abs)
# print(np.max(x_abs_ls,axis=0))
# print(np.sum(x_abs_ls,axis=0))

# import rtde_control
# import rtde_receive
# import time
# import util
#
# rtde_r = rtde_receive.RTDEReceiveInterface("192.168.218.2")
# actual_q = rtde_r.getActualQ()
# actual_pos = rtde_r.getActualTCPPose()
# print(actual_pos)
# actualPosRPY = util.rv2rpy(actual_pos[3],actual_pos[4],actual_pos[5])
# print(type(actualPosRPY))

# rtde_c = rtde_control.RTDEControlInterface("192.168.218.2")
# rtde_r = rtde_receive.RTDEReceiveInterface("192.168.218.2")
# init_q = rtde_r.getActualQ()

# init_q = rtde_r.getActualQ()

# # Target in the robot base
# new_q = init_q[:]
# new_q[0] += 0.20

# # Move asynchronously in joint space to new_q, we specify asynchronous behavior by setting the async parameter to
# # 'True'. Try to set the async parameter to 'False' to observe a default synchronous movement, which cannot be stopped
# # by the stopJ function due to the blocking behaviour.
# rtde_c.moveJ(new_q, 1.05, 1.4, True)
# time.sleep(0.2)
# # Stop the movement before it reaches new_q
# rtde_c.stopJ(0.5)

# # Target in the Z-Axis of the TCP
# target = rtde_r.getActualTCPPose()
# target[2] += 0.10

# # Move asynchronously in cartesian space to target, we specify asynchronous behavior by setting the async parameter to
# # 'True'. Try to set the async parameter to 'False' to observe a default synchronous movement, which cannot be stopped
# # by the stopL function due to the blocking behaviour.
# rtde_c.moveL(target, 0.25, 0.5, True)
# time.sleep(0.2)
# # Stop the movement before it reaches target
# rtde_c.stopL(0.5)

# # Move back to initial joint configuration
# rtde_c.moveJ(init_q)

# # Stop the RTDE control script
# rtde_c.stopScript()
# import numpy as np
# n = np.zeros(48)
# print(type(n))
import  numpy as np
a = np.mat(([1,2,3],[4,5,6]))
b = np.transpose(a).tolist()
print(b)