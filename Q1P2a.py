import matplotlib.pyplot as plt
from math import ceil, floor
import numpy as np
from scipy.stats import norm, multivariate_normal

np.set_printoptions(suppress=True)

plt.rc('font', size=22)
plt.rc('axes', titlesize=18)  #
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=16)
plt.rc('figure', titlesize=22)

np.random.seed(7)

N_train = [20, 200, 2000]
N_valid = 10000

mu = np.array([[2, 2]])

internal_mu = np.array([
    [3, 0], [0, 3],
])

Sigma = np.array([
    [
        [1, 0],
        [0, 1]
    ]
])

internal_Sigma = np.array([
    [
        [2, 0],
        [0, 1]
    ],
    [
        [1, 0],
        [0, 2]
    ]
])

n = mu.shape[1]
priors = np.array([0.65, 0.35])
C = len(priors)

n = mu.shape[1]

X_T = [np.zeros([N_train[0], n]), np.zeros([N_train[1], n]), np.zeros([N_train[2], n])]
y_T = [np.zeros(N_train[0]), np.zeros(N_train[1]), np.zeros(N_train[2])]

X_train = []
y_train = []
Ny_train = []

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
t = 0
for index in range(len(X_T)):
    u = np.random.rand(N_train[index])
    thresholds = np.cumsum(priors)

    ax[floor(t / 2), t % 2].set_title("Training N={} data".format(N_train[index]))

    for c in range(C):
        c_ind = np.argwhere(u <= thresholds[c])[:, 0]
        c_N = len(c_ind)
        y_T[index][c_ind] = c * np.ones(c_N)
        u[c_ind] = 1.1 * np.ones(c_N)
        if (c == 0):
            u = np.random.rand(c_N)
            for num, ind_val in enumerate(c_ind):
                if (u[num] >= 0.5):
                    X_T[index][ind_val] = multivariate_normal.rvs(internal_mu[0], internal_Sigma[0], 1)
                else:
                    X_T[index][ind_val] = multivariate_normal.rvs(internal_mu[1], internal_Sigma[1], 1)

        else:
            X_T[index][c_ind] = multivariate_normal.rvs(mu[0], Sigma[0], c_N)

    X_t = np.column_stack((np.ones([N_train[index]]), X_T[index]))

    n = X_t.shape[1]

    Ny_t = np.array((sum(y_T[index] == 0), sum(y_T[index] == 1)))
    n = X_t.shape[1]

    X_train.append(X_t)
    y_train.append(y_T[index])
    Ny_train.append(Ny_t)
    ax[floor(t / 2), t % 2].scatter(X_t[:, 1], X_t[:, 2], c=y_T[index])
    ax[floor(t / 2), t % 2].set_ylabel("y-axis")
    ax[floor(t / 2), t % 2].set_aspect('equal')
    t += 1

X_valid = np.zeros([N_valid, mu.shape[1]])
y_valid = np.zeros(N_valid)

u = np.random.rand(N_valid)
thresholds = np.cumsum(priors)
for c in range(C):
    c_ind = np.argwhere(u <= thresholds[c])[:, 0]
    c_N = len(c_ind)
    y_valid[c_ind] = c * np.ones(c_N)
    u[c_ind] = 1.1 * np.ones(c_N)
    if (c == 0):
        u = np.random.rand(c_N)
        for num, ind_val in enumerate(c_ind):
            if (u[num] >= 0.5):
                X_valid[ind_val] = multivariate_normal.rvs(internal_mu[0], internal_Sigma[0], 1)
            else:
                a = multivariate_normal.rvs(internal_mu[1], internal_Sigma[1], 1)
                X_valid[ind_val] = multivariate_normal.rvs(internal_mu[1], internal_Sigma[1], 1)

    else:
        X_valid[c_ind] = multivariate_normal.rvs(mu[0], Sigma[0], c_N)

X_valid = np.column_stack((np.ones(N_valid), X_valid))
Ny_valid = np.array((sum(y_valid == 0), sum(y_valid == 1)))

ax[1, 1].set_title("Validation N={} data".format(N_valid))
ax[1, 1].scatter(X_valid[:, 1], X_valid[:, 2], c=y_valid)
ax[1, 1].set_ylabel("y-axis")
ax[1, 1].set_aspect('equal')

x1_valid_lim = (floor(np.min(X_valid[:, 1])), ceil(np.max(X_valid[:, 1])))
x2_valid_lim = (floor(np.min(X_valid[:, 2])), ceil(np.max(X_valid[:, 2])))

plt.setp(ax, xlim=x1_valid_lim, ylim=x2_valid_lim)
plt.tight_layout()
plt.show()

fig;

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def predict_prob(X, theta):
    logits = X.dot(theta)
    return sigmoid(logits)

def log_reg_loss(theta, X, y):
    B = X.shape[0]

    predictions = predict_prob(X, theta)

    error = predictions - y
    nll = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

    g = (1 / B) * X.T.dot(error)

    return nll, g

opts = {}
opts['max_epoch'] = 1000
opts['alpha'] = 1e-3
opts['tolerance'] = 1e-3

opts['batch_size'] = 10


# Breaks the matrix X and vector y into batches
def batchify(X, y, batch_size, N):
    X_batch = []
    y_batch = []

    for i in range(0, N, batch_size):
        nxt = min(i + batch_size, N + 1)
        X_batch.append(X[i:nxt, :])
        y_batch.append(y[i:nxt])

    return X_batch, y_batch


def gradient_descent(loss_func, theta0, X, y, N, *args, **kwargs):
    max_epoch = kwargs['max_epoch'] if 'max_epoch' in kwargs else 200
    alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.1
    epsilon = kwargs['tolerance'] if 'tolerance' in kwargs else 1e-6

    batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 10

    X_batch, y_batch = batchify(X, y, batch_size, N)
    num_batches = len(y_batch)
    print("%d batches of size %d\n" % (num_batches, batch_size))

    theta = theta0
    m_t = np.zeros(theta.shape)

    trace = {}
    trace['loss'] = []
    trace['theta'] = []

    for epoch in range(1, max_epoch + 1):

        loss_epoch = 0
        for b in range(num_batches):
            X_b = X_batch[b]
            y_b = y_batch[b]

            loss, gradient = loss_func(theta, X_b, y_b, *args)
            loss_epoch += loss

            theta = theta - alpha * gradient

            if np.linalg.norm(gradient) < epsilon:
                print("Gradient Descent has converged after {} epochs".format(epoch))
                break

        trace['loss'].append(np.mean(loss_epoch))

        if np.linalg.norm(gradient) < epsilon:
            break

    return theta, trace
def quadratic_transformation(X):
    n = X.shape[1]
    phi_X = X

    phi_X = np.column_stack((phi_X, X[:, 1] * X[:, 1], X[:, 1] * X[:, 2], X[:, 2] * X[:, 2]))

    return phi_X
bounds_X = np.array((floor(np.min(X_valid[:, 1])), ceil(np.max(X_valid[:, 1]))))
bounds_Y = np.array((floor(np.min(X_valid[:, 2])), ceil(np.max(X_valid[:, 2]))))


def create_prediction_score_grid(theta, poly_type='L'):
    xx, yy = np.meshgrid(np.linspace(bounds_X[0], bounds_X[1], 200), np.linspace(bounds_Y[0], bounds_Y[1], 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_aug = np.column_stack((np.ones(200 * 200), grid))
    if poly_type == 'Q':
        grid_aug = quadratic_transformation(grid_aug)
    Z = predict_prob(grid_aug, theta).reshape(xx.shape)

    return xx, yy, Z


def plot_prediction_contours(X, theta, ax, poly_type='L'):
    xx, yy, Z = create_prediction_score_grid(theta, poly_type)
    cs = ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.55)
    ax.set_xlim([bounds_X[0], bounds_X[1]])
    ax.set_ylim([bounds_Y[0], bounds_Y[1]])


def plot_decision_boundaries(X, labels, theta, ax, poly_type='L'):
    ax.plot(X[labels == 0, 1], X[labels == 0, 2], 'o', label="Class 0")
    ax.plot(X[labels == 1, 1], X[labels == 1, 2], '+', label="Class 1")

    xx, yy, Z = create_prediction_score_grid(theta, poly_type)
    cs = ax.contour(xx, yy, Z, levels=1, colors='k')

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect('equal')


def report_logistic_classifier_results(X, theta, labels, N_labels, ax, poly_type='L'):
    predictions = predict_prob(X, theta)
    decisions = np.array(predictions >= 0.5)

    ind_00 = np.argwhere((decisions == 0) & (labels == 0))
    tnr = len(ind_00) / N_labels[0]
    ind_10 = np.argwhere((decisions == 1) & (labels == 0))
    fpr = len(ind_10) / N_labels[0]
    ind_01 = np.argwhere((decisions == 0) & (labels == 1))
    fnr = len(ind_01) / N_labels[1]
    ind_11 = np.argwhere((decisions == 1) & (labels == 1))
    tpr = len(ind_11) / N_labels[1]

    prob_error = fpr * priors[0] + fnr * priors[1]

    print("The total error achieved with this classifier is {:.3f}".format(prob_error))

    ax.plot(X[ind_00, 1], X[ind_00, 2], 'og', label="Class 0 Correct", alpha=.25)
    ax.plot(X[ind_10, 1], X[ind_10, 2], 'or', label="Class 0 Wrong")
    ax.plot(X[ind_01, 1], X[ind_01, 2], '+r', label="Class 1 Wrong")
    ax.plot(X[ind_11, 1], X[ind_11, 2], '+g', label="Class 1 Correct", alpha=.25)

    # Draw the decision boundary based on whether its linear (L) or quadratic (Q)
    plot_prediction_contours(X, theta, ax, poly_type)
    ax.set_aspect('equal')
theta0_linear = np.random.randn(n)

fig_decision, ax_decision = plt.subplots(3, 2, figsize=(15, 15));

print("Training the logistic-linear model with GD per data subset"),
for i in range(len(N_train)):
    shuffled_indices = np.random.permutation(N_train[i])
    X = X_train[i][shuffled_indices]
    y = y_train[i][shuffled_indices]

    theta_gd, trace = gradient_descent(log_reg_loss, theta0_linear, X, y, N_train[i], **opts)

    print("Logistic-Linear N={} GD Theta: {}".format(N_train[i], theta_gd))
    print("Logistic-Linear N={} NLL: {}".format(N_train[i], trace['loss'][-1]))
    theta_hist = np.asarray(trace['theta'])
    nll_hist = np.array(trace['loss'])

    plot_decision_boundaries(X, y, theta_gd, ax_decision[i, 0])
    ax_decision[i, 0].set_title("Decision Boundary for \n Logistic-Linear Model N={}".format(X.shape[0]))

    report_logistic_classifier_results(X_valid, theta_gd, y_valid, Ny_valid, ax_decision[i, 1])
    ax_decision[i, 1].set_title(
        "Classifier Decisions on Validation Set \n Logistic-Linear Model N={}".format(N_train[i]))

plt.setp(ax_decision, xlim=x1_valid_lim, ylim=x2_valid_lim)

plt.subplots_adjust(left=0.05,
                    bottom=0.05,
                    right=0.6,
                    top=0.95,
                    wspace=0.1,
                    hspace=0.3)

handles, labels = ax_decision[0, 1].get_legend_handles_labels()
fig_decision.legend(handles, labels, loc='lower center')

plt.show()
