# %% Library
import numpy as np
from PIL import Image
# from matplotlib import pylab as plt
import matplotlib.pyplot as plt
import csv

# %% Calibration
clb = 20/1 # (px/mm)

# %% data import
file_name = "./Results.txt"
data = np.loadtxt(file_name, delimiter="\t", skiprows=1, dtype='float')

image_name = "./sample.bmp"
im = np.array(Image.open(image_name))

# Show image 
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
plt.imshow(im)

# Print type & size
print("Type is ", im.dtype)
print("Size is ", im.shape)
x_max = im.shape[0]
y_max = im.shape[1]

# %% X- and Y-coordinate
X = data[:, 5]
Y = data[:, 6]

# %%
# Number of drop to be analyzed
n_drop = np.size(X)/5 # 5 -> 5-point approximation
n_drop = int(n_drop)

# 
output_data = np.zeros((n_drop,8))

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.invert_yaxis()

# Main loop
cnt = 0
for n in range(n_drop):
    
    M = np.zeros((5,5))
    b = np.zeros((5,1))
    
    for i in range(5):
        
        k = i + 5*n
        
        M[0, 0] += X[k]*X[k]*Y[k]*Y[k]
        M[0, 1] += X[k]*Y[k]*Y[k]*Y[k]
        M[0, 2] += X[k]*X[k]*Y[k]
        M[0, 3] += X[k] * Y[k]*Y[k]
        M[0, 4] += X[k] * Y[k]
        M[1, 0] += X[k]*Y[k]*Y[k]*Y[k]
        M[1, 1] += Y[k]*Y[k]*Y[k]*Y[k]
        M[1, 2] += X[k] * Y[k]*Y[k]
        M[1, 3] += Y[k]*Y[k]*Y[k]
        M[1, 4] += Y[k]*Y[k]
        M[2, 0] += X[k]*X[k]*Y[k]
        M[2, 1] += X[k] * Y[k]*Y[k]
        M[2, 2] += X[k]*X[k]
        M[2, 3] += X[k] * Y[k]
        M[2, 4] += X[k]
        M[3, 0] += X[k] * Y[k]*Y[k]
        M[3, 1] += Y[k]*Y[k]*Y[k]
        M[3, 2] += X[k] * Y[k]
        M[3, 3] += Y[k]*Y[k]
        M[3, 4] += Y[k]
        M[4, 0] += X[k] * Y[k]
        M[4, 1] += Y[k]*Y[k]
        M[4, 2] += X[k]
        M[4, 3] += Y[k]
        M[4, 4] += 1
        b[0, 0] -= X[k]*X[k]*X[k]*Y[k]
        b[1, 0] -= X[k]*X[k]*Y[k]*Y[k]
        b[2, 0] -= X[k]*X[k]*X[k]
        b[3, 0] -= X[k]*X[k]*Y[k]
        b[4, 0] -= X[k]*X[k]
    
    Minv = np.linalg.inv(M)
    a = np.matmul(Minv,b)
    
    # 
    A = a[0]
    B = a[1]
    C = a[2]
    D = a[3]
    E = a[4]
    
    # 
    X0 = (A*D - 2*B*C)/(4*B - A*A)
    Y0 = (A*C - 2*D)/(4*B - A*A)
    θ = 0.5*np.arctan(A/(1-B))
    
    # 
    aa = np.sqrt((X0*np.cos(θ)+Y0*np.sin(θ)) ** 2 - E*(np.cos(θ)) ** 2 - ((X0*np.sin(θ) - Y0*np.cos(θ)) ** 2 - E*(np.sin(θ)) ** 2) * ((np.sin(θ)) ** 2 - B * (np.cos(θ)) ** 2)/((np.cos(θ)) ** 2 - B*(np.sin(θ)) ** 2))
    bb = np.sqrt((X0*np.sin(θ)-Y0*np.cos(θ)) ** 2 - E*(np.sin(θ)) ** 2 - ((X0*np.cos(θ) + Y0*np.sin(θ)) ** 2 - E*(np.cos(θ)) ** 2) * ((np.cos(θ)) ** 2 - B * (np.sin(θ)) ** 2)/((np.sin(θ)) ** 2 - B*(np.cos(θ)) ** 2))
    # print([X0,Y0])
    
    # Output
    output_data[cnt, 0] = X0                    # x-centroid in px
    output_data[cnt, 1] = Y0                    # y-centroid in px
    output_data[cnt, 2] = aa                    # a-axis in px
    output_data[cnt, 3] = bb                    # b-axis in px
    output_data[cnt, 4] = θ                     # lean angle
    output_data[cnt, 5] = np.pi*aa*bb           # area in px^2
    output_data[cnt, 6] = np.pi*aa*bb/(clb*clb) # area in mm^2
    output_data[cnt, 7] = 2*np.sqrt(aa*bb/clb/clb) # area equivalent diameter in mm
    
    # Counter
    cnt += 1
    
    # meshgrid
    xx, yy = np.meshgrid(range(x_max), range(y_max))
    xx,yy = xx+1,yy+1
    # Contour
    V = xx ** 2 + A * xx*yy + B*yy*yy + C*xx + D*yy + E
    # print
    
    # plt.hold(True)
    ax.contour(xx, yy, V, levels=[0])
    ax.text(X0, Y0, str(n+1))
    #


    
# %% Data save
with open('dropInfo.csv', 'w', newline="") as csv_file:
    # header setting
    fieldnames = ['X0', 'Y0', 'a', 'b', 'theta', 'area [px^2]', 'area [mm^2]', 'Deq [mm]']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Writing data
    for n in range(n_drop):
        writer.writerow({
            'X0': output_data[n, 0], 
            'Y0': output_data[n, 1], 
            'a': output_data[n, 2], 
            'b': output_data[n, 3], 
            'theta': output_data[n, 4],
            'area [px^2]': output_data[n, 5],
            'area [mm^2]': output_data[n, 6],
            'Deq [mm]': output_data[n, 7]})

# %%
