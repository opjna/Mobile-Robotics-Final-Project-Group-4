import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import scipy.stats as stats

from scipy.optimize import minimize
from tskew.tskew import getObjectiveFunction, tspdf_1d

import pyreadr

esti_delta_r = np.load('../../data/deltar_sift.npy')
true_delta_r = np.load('../../data/deltar_tru_sift.npy')

error = esti_delta_r - true_delta_r

fig, axs = plt.subplots(1, 2, sharex=True)
fig.suptitle('True and estimated frame-to-frame displacements')

axs[0].plot(true_delta_r, label=r'True $\Delta r$', linewidth=5)
axs[0].plot(esti_delta_r,linestyle='--', label=r'Est. $\Delta r$', linewidth=5)
axs[0].set_xlabel('Frame index')
axs[0].set_ylabel('Distance (m)')
axs[0].set_title('Overlayed')
axs[0].legend()

axs[1].plot(error)
axs[1].set_xlabel('Frame index')
axs[1].set_ylabel('Error (m)')
axs[1].set_title('Difference')

plt.figure()
stats.probplot(error, dist="norm", plot=plt)
plt.title('Probability Plot for Visual Odometry Errors')
plt.show()


realization = error
loc = np.mean(realization)
scale = np.var(realization)
df = 10
skew = 0

theta = np.array([loc, scale, df, skew])

res = minimize(getObjectiveFunction(realization, use_loglikelihood=True), x0=theta,
               method='Nelder-Mead')

N = 1_000
xmin = np.min(realization)
xmax = np.max(realization)
extent =  xmax - xmin
xvals = np.linspace(xmin - 0.1 * extent, xmax + 0.1 * extent, N)

plt.figure()
est_pdf = tspdf_1d(xvals, res.x[0], res.x[1], res.x[2], res.x[3])
plt.hist(realization, bins=100, density=True, color='green', alpha=0.5)
plt.plot(xvals, est_pdf, linestyle='--', label='Estimated skew t', linewidth=3, alpha=0.5)
plt.xlim([xmin, xmax])
plt.title(r'Modeling Visual Odometry Errors using Skew-$t$ Distribution')
plt.legend()



