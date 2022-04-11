import numpy as np
from generalized_unscented_transform import generalized_ut
from evaluate_from_sigma_points import Evaluate_sample_statistics


def quadratic_transform(sigma_points):
    sigma_points = np.array(sigma_points)[0]
    y = []
    for x in sigma_points:
        y.append(3*x + 2*(x**2))
    y = np.matrix(y)
    return y


# Gaussian case
def test_Gaussian():
    m1 = 1
    m2 = 4
    m3 = 0
    m4 = 48
    sigma_points, weights = generalized_ut(np.matrix([m1]), np.matrix([m2]), np.matrix([m3]), np.matrix([m4]))
    sigma_mean, sigma_cov, sigma_skew, sigma_kurt = Evaluate_sample_statistics(quadratic_transform(sigma_points), weights)
    print(sigma_mean)
    print(sigma_cov)
    print(sigma_skew)
    print(sigma_kurt)

def test_Exp():
    m1 = 0.5
    m2 = 0.25
    m3 = 0.25
    m4 = 0.5625
    sigma_points, weights = generalized_ut(np.matrix([m1]), np.matrix([m2]), np.matrix([m3]), np.matrix([m4]))
    sigma_mean, sigma_cov, sigma_skew, sigma_kurt = Evaluate_sample_statistics(quadratic_transform(sigma_points), weights)
    print(sigma_mean)
    print(sigma_cov)
    print(sigma_skew)
    print(sigma_kurt)


if __name__ == "__main__":
    test_Exp()
