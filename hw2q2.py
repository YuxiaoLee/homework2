from math import floor

import numpy as np
from numpy import ceil

from data_generation import*

def hw2q2():
    Ntrain = 100
    data = generateData(Ntrain)
    plot3(data[0,:],data[1,:],data[2,:])
    xTrain = data[0:2,:]
    yTrain = data[2,:]

    Ntrain = 1000
    data = generateData(Ntrain)
    plot3(data[0,:],data[1,:],data[2,:])
    xValidate = data[0:2,:]
    yValidate = data[2,:]

    return xTrain,yTrain,xValidate,yValidate

def generateData(N):
    gmmParameters = {}
    gmmParameters['priors'] = [.3,.4,.3] # priors should be a row vector
    gmmParameters['meanVectors'] = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])
    gmmParameters['covMatrices'] = np.zeros((3, 3, 3))
    gmmParameters['covMatrices'][:,:,0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][:,:,1] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][:,:,2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    x,labels = generateDataFromGMM(N,gmmParameters)
    return x

def generateDataFromGMM(N,gmmParameters):
#    Generates N vector samples from the specified mixture of Gaussians
#    Returns samples and their component labels
#    Data dimensionality is determined by the size of mu/Sigma parameters
    priors = gmmParameters['priors'] # priors should be a row vector
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0] # Data dimensionality
    C = len(priors) # Number of components
    x = np.zeros((n,N))
    labels = np.zeros((1,N))
    # Decide randomly which samples will come from each component
    u = np.random.random((1,N))
    thresholds = np.zeros((1,C+1))
    thresholds[:,0:C] = np.cumsum(priors)
    thresholds[:,C] = 1
    for l in range(C):
        indl = np.where(u <= float(thresholds[:,l]))
        Nl = len(indl[1])
        labels[indl] = (l+1)*1
        u[indl] = 1.1
        x[:,indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[:,l], covMatrices[:,:,l], Nl))

    return x,labels

def plot3(a,b,c,mark="o",col="b"):
  from matplotlib import pyplot
  import pylab
  from mpl_toolkits.mplot3d import Axes3D
  pylab.ion()
  fig = pylab.figure()
  ax = Axes3D(fig)
  ax.scatter(a, b, c,marker=mark,color=col)
  ax.set_xlabel("x1")
  ax.set_ylabel("x2")
  ax.set_zlabel("y")
  ax.set_title('Training Dataset')


X_train, y_train = generate_q2_data(100, "Training")
X_valid, y_valid = generate_q2_data(1000, "Validation")


def mle_solution(X, y):
    # ML parameter solution is (X^T*X)^-1 * X^T * y
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


def map_solution(X, y, gamma):
    # MAP parameter solution is (X^T*X + (1/gamma)*I)^-1 * X^T * y
    return np.linalg.inv(X.T.dot(X) + (1 / gamma)*np.eye(X.shape[1])).dot(X.T).dot(y)


def mse(y_preds, y_true):
    # Residual error (X * theta) - y
    error = y_preds - y_true
    # Return MSE
    return np.mean(error ** 2)


# First apply the cubic transformation of the inputs to the training set
phi = PolynomialFeatures(degree=3)
X_train_cubic = phi.fit_transform(X_train)

# Derive the MLE parameter solution
theta_mle = mle_solution(X_train_cubic, y_train)

# Then produce predictions on the validation samples using this ML estimator
X_valid_cubic = phi.transform(X_valid)
y_pred_mle = X_valid_cubic.dot(theta_mle)

mse_mle = mse(y_pred_mle, y_valid)
print("MSE on Validation set for ML parameter estimator: %.3f" % mse_mle)

x1_valid_lim = (floor(np.min(X_valid[:,0])), ceil(np.max(X_valid[:,0])))
x2_valid_lim = (floor(np.min(X_valid[:,1])), ceil(np.max(X_valid[:,1])))
# Regression model prediction function to feed into grid creation routine
reg_fun = lambda X, th: X.dot(th)
xx, yy, Z = create_prediction_score_grid(x1_valid_lim, x2_valid_lim, theta_mle, reg_fun, phi, num_coords=100)


fig_mle = plt.figure(figsize=(10, 10))
ax_mle = plt.axes(projection ='3d')

# Plot the best fit plane on the 2D real vector samples
ax_mle.scatter(X_valid[:,0], X_valid[:,1], y_valid, marker='o', color='b');
ax_mle.plot_surface(xx, yy, Z, color='red', alpha=0.3);
ax_mle.set_xlabel(r"$x_1$")
ax_mle.set_ylabel(r"$x_2$")
ax_mle.set_zlabel(r"$y$")

# To set the axes equal for a 3D plot
ax_mle.set_box_aspect((np.ptp(X_valid[:,0]), np.ptp(X_valid[:,1]), np.ptp(y_valid)))

plt.title("The Estimator of Validation Set ML")
plt.tight_layout()
plt.show()


# Use geomspace to return 1000 evenly spaced numbers on a log scale with start-end points specified
n_gammas = 1000
# Gammas in the range 10^-7 to 10^7
gammas = np.geomspace(10**-7, 10**7, num=n_gammas)
mse_map = np.empty(n_gammas)
for i, gamma in enumerate(gammas):
    theta_map = map_solution(X_train_cubic, y_train, gamma)
    y_pred_map = X_valid_cubic.dot(theta_map)
    mse_map[i] = mse(y_pred_map, y_valid)

print(r"Best MSE on Validation set for MAP is: %.3f" % np.min(mse_map))
print(r"Gamma of the best MSE on Validation set for MAP is %f" % gammas[np.argmin(mse_map)])
# Plot MSE vs regularizer parameter gamma
fig_map, ax_map = plt.subplots(figsize=(10, 10))
ax_map.plot(gammas, mse_map, color='b', label=r"$\theta_{MAP}$")
plt.axhline(y=mse_mle, xmin=10**-7, xmax=10**7, color='red', label=r"$\theta_{MLE}$")

ax_map.set_xscale('log')
ax_map.set_xticks(np.geomspace(10**-7, 10**7, num=15))

ax_map.set_xlabel(r"$\gamma$")
ax_map.set_ylabel(r"$MSE_{valid}$")
ax_map.set_title("The Estimator of Validation Set MAP")

plt.legend()
plt.show()
# Reference from MARK ZOLOTAS
