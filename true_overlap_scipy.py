import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def get_true_overlap(mu1, mu2, covariance, min, max, step, threshold):
    xs, ys = np.mgrid[min:max:step, min:max:step]
    ts = np.dstack((xs, ys))
    f1 = stats.multivariate_normal.pdf(x=ts, mean=mu1, cov=covariance, allow_singular=True)
    f2 = stats.multivariate_normal.pdf(x=ts, mean=mu2, cov=covariance, allow_singular=True)

    overlap = (f1 > threshold) & (f2 > threshold)
    grid = np.dstack((xs, ys))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xs, ys, f1, cmap='viridis', linewidth=0)
    ax.plot_surface(xs, ys, f2, cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

    return grid[overlap], overlap

def check_true_overlap(mu1, mu2, covariance1, covariance2, x1, x2, threshold, dim):
    X = np.concatenate([x1, x2]).reshape(-1, dim)
    ts = X
    f1 = stats.multivariate_normal.pdf(x=ts, mean=mu1, cov=covariance1, allow_singular=True)
    f2 = stats.multivariate_normal.pdf(x=ts, mean=mu2, cov=covariance2, allow_singular=True)

    overlap = (f1 > threshold) & (f2 > threshold)

    return overlap

def check_true_overlap_2_regions(mu1, mu2, covariance1, covariance2, x1, x2, threshold, dim):
    X = np.concatenate([x1, x2]).reshape(-1, dim)
    ts = X
    f1 = stats.multivariate_normal.pdf(x=ts, mean=mu1, cov=covariance1, allow_singular=True)
    f2 = stats.multivariate_normal.pdf(x=ts, mean=mu2, cov=covariance2, allow_singular=True)
    f3 = stats.multivariate_normal.pdf(x=ts, mean=-mu2, cov=covariance2, allow_singular=True)

    overlap = (f1 > threshold) & ((f2 + f3)/2.0 > threshold)

    return overlap

def get_true_overlap_2_regions(mu1, mu2, covariance1, covariance2, min, max, step, threshold):
    xs, ys = np.mgrid[min:max:step, min:max:step]
    ts = np.dstack((xs, ys))
    f1 = stats.multivariate_normal.pdf(x=ts, mean=mu1, cov=covariance1, allow_singular=True)
    f2 = stats.multivariate_normal.pdf(x=ts, mean=mu2, cov=covariance2, allow_singular=True)
    f3 = stats.multivariate_normal.pdf(x=ts, mean=-mu2, cov=covariance2, allow_singular=True)

    overlap = (f1 > threshold) & ((f2 + f3)/2.0 > threshold)
    grid = np.dstack((xs, ys))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xs, ys, f1, cmap='viridis', linewidth=0)
    ax.plot_surface(xs, ys, (f2+f3)/2.0, cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()

    return grid[overlap], overlap

def check_true_overlap_2_regions_uniform(l1, r1, mu2, covariance2, x1, x2, threshold, dim):
    X = np.concatenate([x1, x2]).reshape(-1, dim)
    ts = X
    uni = stats.uniform.pdf(x=ts, loc=l1, scale=(r1-l1))
    # multiply columns
    f1 = uni[:,0]
    for i in range(1, dim):
        f1 = f1 * uni[:,i]
    f2 = stats.multivariate_normal.pdf(x=ts, mean=mu2, cov=covariance2, allow_singular=True)
    f3 = stats.multivariate_normal.pdf(x=ts, mean=-mu2, cov=covariance2, allow_singular=True)
    np.set_printoptions(threshold=np.inf)
    overlap = (f1 > threshold) & ((f2 + f3)/2.0 > threshold)

    return overlap

def check_true_overlap_2_regions_separate(mu1, mu2, covariance1, covariance2, x1, x2, threshold, dim):
    X = np.concatenate([x1, x2]).reshape(-1, dim)
    ts = X
    f1 = stats.multivariate_normal.pdf(x=ts, mean=mu1, cov=covariance1, allow_singular=True)
    f2 = stats.multivariate_normal.pdf(x=ts, mean=mu2, cov=covariance2, allow_singular=True)
    f3 = stats.multivariate_normal.pdf(x=ts, mean=-mu1, cov=covariance1, allow_singular=True)
    f4 = stats.multivariate_normal.pdf(x=ts, mean=-mu2, cov=covariance2, allow_singular=True)
    # each class consists of 2 gaussians, density should be divided by 2, or threshold * 2
    eps = threshold * 2.0

    overlap = (f1 > eps) & (f2 > eps) | (f3 > eps) & (f4> eps)

    return overlap