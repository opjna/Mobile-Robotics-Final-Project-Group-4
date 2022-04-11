import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import scipy.stats as stats

from scipy.optimize import minimize
from tskew.tskew import getObjectiveFunction, tspdf_1d
from tskew.tskew import ts_invcdf

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
#
# res = minimize(getObjectiveFunction(realization, use_loglikelihood=True), x0=theta,
#                method='Nelder-Mead')

N = 1_000
xmin = -0.05
xmax = 0.05
extent =  xmax - xmin
xvals = np.linspace(xmin - 0.1 * extent, xmax + 0.1 * extent, N)

# plt.figure()
# est_pdf = tspdf_1d(xvals, res.x[0], res.x[1], res.x[2], res.x[3])
# plt.hist(realization, bins=500, density=True, color='green', alpha=0.5)
# plt.plot(xvals, est_pdf, linestyle='--', label='Estimated skew t', linewidth=3, alpha=0.5)
# plt.xlim([xmin, xmax])
# plt.legend()

loc = 2.72e-5
scale = 2.25e-6
df = 1
skew = 2.8e-3

# median = ts_invcdf(np.array([0.25, 0.5, 0.75]), loc, scale, df, skew)