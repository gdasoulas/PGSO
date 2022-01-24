import numpy as np
import matplotlib.pyplot as plt 

workspace = './res'
# workspace = 'new_exp/'
p = np.loadtxt(workspace+'p_values.out')
q = np.loadtxt(workspace+'q_values.out')
r = np.loadtxt(workspace+'r_values.out')
c1 = np.loadtxt(workspace+'c1_values.out')
c2 = np.loadtxt(workspace+'c2_values.out')
c3 = np.loadtxt(workspace+'c3_values.out')
d1 = np.loadtxt(workspace+'d1_values.out')
degrees = np.loadtxt(workspace+'degrees_values.out')
cond = np.loadtxt(workspace+'cond_values.out')

centers, radii = [], []
for i in range(len(p)):
    tmp1 = degrees+d1[i,:]
    tmp2 = c3[i]*tmp1**p[i]
    tmp3 = c2[i]*tmp1**(q[i]+r[i])
    tmp4 = tmp3 * d1[i,:]
    center = tmp2 + tmp4 + c1[i]
    centers.append(center)
    radii.append(np.abs(c2[i])*(tmp1)**(q[i]+r[i])*degrees)


print(len(centers))

bounds = []
for i in range(len(p)):
    min_bound = min(centers[i] - radii[i])
    max_bound = max(centers[i] + radii[i])
    bounds.append([min_bound, max_bound])
np.savetxt(workspace+'centers_gersh.out', centers)
np.savetxt(workspace+'radii_gersh.out', radii)
np.savetxt(workspace+'bounds.out', bounds)


# STANDARD GCN
centers, radii = [], []
for i in range(len(p)):
    tmp1 = degrees + 1
    tmp3 = 1*tmp1**(-1)
    tmp4 = tmp3
    center = tmp4

    centers.append(-1/(degrees+1))
    radii.append(degrees/(degrees+1))
    # centers.append(center)
    radii.append((tmp1)**(-1)*degrees)

gcn_bounds = []
for i in range(len(p)):
    min_bound = min(centers[i] - radii[i])
    max_bound = max(centers[i] + radii[i])
    gcn_bounds.append([min_bound, max_bound])

print(gcn_bounds[0])
epochs = range(len(p))
for i in range(len(p)):
    plt.vlines(i, bounds[i][0], bounds[i][1] )
    plt.vlines(i+0.2, gcn_bounds[i][0], gcn_bounds[i][1], color='r' )

# plt.plot(epochs, cond)
plt.xlabel('Epochs')
plt.ylabel('Spectral bounds')
# plt.title('Spectral support of $L_{gen}$')
plt.savefig(workspace+'spectral_support_Lgen.pdf')

# exit()


sorted_cond= np.sort(cond)
print(sorted_cond)
cond[cond > sorted_cond[50] ] = sorted_cond[50]

plt.figure()
plt.plot(epochs, cond)
plt.xlabel('Epochs')
plt.ylabel('Condition Number')
# plt.title('Condition number of $L_{gen}')
plt.savefig(workspace+'condition_number_Lgen.pdf')

plt.figure()
plt.plot(epochs[20:], cond[20:])
plt.xlabel('Epochs')
plt.ylabel('Condition Number')
# plt.title('Condition number of $L_{gen}')
plt.savefig(workspace+'condition_number_Lgen_20.pdf')

plt.figure()
plt.plot(epochs[35:50], cond[35:50])
plt.xlabel('Epochs')
plt.ylabel('Condition Number')
# plt.title('Condition number of $L_{gen}')
plt.savefig(workspace+'condition_number_Lgen_35.pdf')

# plt.show()

