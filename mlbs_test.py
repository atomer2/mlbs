import numpy as np
from mlbs import mlbs


data = np.loadtxt('approx_data.txt')
start_res = np.array([2, 2, 2, 2, 3])
target_res = np.array([50, 50, 50, 50, 3])
input_range = np.array([[0, 0, 0, 0], [100.1, 100.1, 100.1, 100.1]])
s = mlbs(data, target_res, start_res,  input_range)
# multilevel B spline approximation
s.approximation()

# test sample interpolation
res = []
test = np.loadtxt('test_data.txt')
for t in test:
    apprx_value = s.slbs.interpolation(t[:4])
    line = np.append(apprx_value, t[4:])
    res.append(line)

res = np.array(res)
np.savetxt('apprx_compare.txt', res, fmt='%10.5f')

# calculate color difference
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
import matplotlib.pyplot as plt

delta_e_arr = []

for i in res:
    calc_color = LabColor(*i[:3])
    real_color = LabColor(*i[3:])
    d = delta_e_cie2000(real_color, calc_color)
    delta_e_arr.append(d)

delta_e_arr = np.array(delta_e_arr)

# plot hist
arr = plt.hist(delta_e_arr, facecolor='green', bins=10, align='mid',
               range=(0, 10), rwidth=0.9, label='MRBS', edgecolor='blue')

for i in range(10):
    plt.text(arr[1][i] + 0.3, arr[0][i] + 5, str(int(arr[0][i])))
plt.legend()
plt.title("color difference")
plt.show()
