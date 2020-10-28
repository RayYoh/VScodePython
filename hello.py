import numpy as np
import xlrd
path=r'D:\1A.研究生\科研\基于力的位姿计算\不同姿态下的传感器输出值.xlsx'  # xls dir
data = xlrd.open_workbook(path)  # read excel
Sheet = data.sheet_by_index(0)  # read sheet
Sheet_rows = Sheet.nrows  # read row
print(Sheet_rows)
vol_Force = []
RPY_angle = []   
for i in range(0, 10):
    data_str = Sheet.row_values(i)  # read by row
    force_list = [float(f) for f in data_str[4:10]]
    vol_Force.append(force_list)  # read force
    angle_list = [float(a) for a in data_str[19:22]]
    RPY_angle.append(angle_list)  # read attitude angle
print(vol_Force)