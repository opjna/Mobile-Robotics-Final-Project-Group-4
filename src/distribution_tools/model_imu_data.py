import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import scipy.stats as stats

from scipy.optimize import minimize
from tskew.tskew import getObjectiveFunction, tspdf_1d


import pyreadr

result = pyreadr.read_r('/home/efvega/data/imu/cont.imu1.rda') # also works for Rds
data = result['cont.imu1']
numpy_data = np.squeeze(data.to_numpy())
# done! let's see what we got
# result is a dictionary where keys are the name of objects and the values python
# objects
print(result.keys()) # let's check what objects we got
# df1 = result["df1"] # extract the pandas data frame for object df1



stats.probplot(numpy_data, dist="norm", plot=plt)
plt.show()


realization = numpy_data
loc = np.mean(realization)
scale = np.var(realization)
df = 1000
skew = 0

theta = np.array([loc, scale, df, skew])

res = minimize(getObjectiveFunction(realization, use_loglikelihood=True), x0=theta,
               method='Nelder-Mead')

N = 1_000
extent =  np.max(realization) - np.min(realization)
xvals = np.linspace(np.min(realization) - 0.1 * extent, np.max(realization) + 0.1 * extent, N)

plt.figure()
est_pdf = tspdf_1d(xvals, res.x[0], res.x[1], res.x[2], res.x[3])
plt.hist(realization, bins=100, density=True)
plt.plot(xvals, est_pdf, linestyle='--', label='Estimated skew t', linewidth=4)
plt.legend()