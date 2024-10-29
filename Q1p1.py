from scipy.stats import multivariate_normal as mvn
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.preprocessing import PolynomialFeatures
from sys import float_info  # Threshold smallest positive floating value

def generate_q1_data(N, pdf_params):
    n = 2

    # Get random samples from a uniform distribution to assign to each class
    u = np.random.rand(N)
    labels = u >= pdf_params['priors'][0]

    X = np.zeros((N, n))
    for i in range(N):
        if labels[i] == 0:
            # Choose randomly from the two Gaussians in Class 0
            if np.random.rand() < pdf_params['gmm_a'][0]:
                X[i, :] = np.random.multivariate_normal(pdf_params['mu'][0], pdf_params['Sigma'][0])
            else:
                X[i, :] = np.random.multivariate_normal(pdf_params['mu'][1], pdf_params['Sigma'][1])
        else:
            # Choose randomly from the two Gaussians in Class 1
            if np.random.rand() < pdf_params['gmm_a'][2]:
                X[i, :] = np.random.multivariate_normal(pdf_params['mu'][2], pdf_params['Sigma'][2])
            else:
                X[i, :] = np.random.multivariate_normal(pdf_params['mu'][3], pdf_params['Sigma'][3])

    return X, labels


def create_prediction_score_grid(bounds_X, bounds_Y, params, prediction_function, phi=None, num_coords=200):
    # Note that I am creating a 200x200 rectangular grid
    xx, yy = np.meshgrid(np.linspace(bounds_X[0], bounds_X[1], num_coords),
                         np.linspace(bounds_Y[0], bounds_Y[1], num_coords))

    # Flattening grid and feed into a fitted transformation function if provided
    grid = np.c_[xx.ravel(), yy.ravel()]
    if phi:
        grid = phi.transform(grid)

    # Z matrix are the predictions given the provided model parameters
    Z = prediction_function(grid, params).reshape(xx.shape)

    return xx, yy, Z


def generate_gmm_data(N, gmm_pdf):
    # Generates N vector samples from the specified mixture of Gaussians
    # Returns samples and their component labels
    # Data dimensionality is determined by the size of mu/Sigma parameters

    # Decide randomly which samples will come from each component
    u = np.random.random(N)
    thresholds = np.cumsum(gmm_pdf['priors'])
    thresholds = np.insert(thresholds, 0, 0)  # For intervals of classes

    n = gmm_pdf['mu'].shape[0]  # Data dimensionality

    X = np.zeros((N, n))
    C = len(gmm_pdf['priors'])  # Number of components
    for i in range(C + 1):
        # Get randomly sampled indices for this Gaussian, checking between thresholds based on class priors
        indices = np.argwhere((thresholds[i - 1] <= u) & (u <= thresholds[i]))[:, 0]
        # No. of samples in this Gaussian
        X[indices, :] = multivariate_normal.rvs(gmm_pdf['mu'][i - 1], gmm_pdf['Sigma'][i - 1], len(indices))

    return X[:, 0:2], X[:, 2]


np.set_printoptions(suppress=True)

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)

plt.rc('font', size=22)          # controls default text sizes
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=18)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('figure', titlesize=22)   # fontsize of the figure title


# Generate ROC curve samples
def estimate_roc(discriminant_score, labels, N_labels):
    # Sorting necessary so the resulting FPR and TPR axes plot threshold probabilities in order as a line
    sorted_score = sorted(discriminant_score)

    # Use gamma values that will account for every possible classification split
    # The epsilon is just to account for the two extremes of the ROC curve (TPR=FPR=0 and TPR=FPR=1)
    gammas = ([sorted_score[0] - float_info.epsilon] +
              sorted_score +
              [sorted_score[-1] + float_info.epsilon])

    # Calculate the decision label for each observation for each gamma
    decisions = [discriminant_score >= g for g in gammas]

    # Retrieve indices where FPs occur
    ind10 = [np.argwhere((d == 1) & (labels == 0)) for d in decisions]
    # Compute FP rates (FPR) as a fraction of total samples in the negative class
    p10 = [len(inds) / N_labels[0] for inds in ind10]
    # Retrieve indices where TPs occur
    ind11 = [np.argwhere((d == 1) & (labels == 1)) for d in decisions]
    # Compute TP rates (TPR) as a fraction of total samples in the positive class
    p11 = [len(inds) / N_labels[1] for inds in ind11]

    # ROC has FPR on the x-axis and TPR on the y-axis, but return others as well for convenience
    roc = {}
    roc['p10'] = np.array(p10)
    roc['p11'] = np.array(p11)

    return roc, gammas


def get_binary_classification_metrics(predictions, labels, N_labels):
    # Get indices and probability estimates of the four decision scenarios:
    # (true negative, false positive, false negative, true positive)
    class_metrics = {}

    # True Negative Probability Rate
    class_metrics['TN'] = np.argwhere((predictions == 0) & (labels == 0))
    class_metrics['TNR'] = len(class_metrics['TN']) / N_labels[0]
    # False Positive Probability Rate
    class_metrics['FP'] = np.argwhere((predictions == 1) & (labels == 0))
    class_metrics['FPR'] = len(class_metrics['FP']) / N_labels[0]
    # False Negative Probability Rate
    class_metrics['FN'] = np.argwhere((predictions == 0) & (labels == 1))
    class_metrics['FNR'] = len(class_metrics['FN']) / N_labels[1]
    # True Positive Probability Rate
    class_metrics['TP'] = np.argwhere((predictions == 1) & (labels == 1))
    class_metrics['TPR'] = len(class_metrics['TP']) / N_labels[1]

    return class_metrics

pdf_params = {
    'priors': np.array([0.6, 0.4]),
    'gmm_a': np.array([0.5, 0.5, 0.5, 0.5]),
    'mu': np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]]),
    'Sigma': np.array([[[1, 0], [0, 1]]] * 4)
}

N_train = [20, 200, 2000]
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

train_data = [generate_q1_data(N, pdf_params) for N in N_train]
X_train, y_train = zip(*train_data)

for i, ax in enumerate(axes.flatten()[:-1]):
    ax.set_title(r"Training $D^{%d}$" % N_train[i])
    ax.plot(X_train[i][y_train[i] == 0, 0], X_train[i][y_train[i] == 0, 1], 'co', label="Class 0")  # 青色
    ax.plot(X_train[i][y_train[i] == 1, 0], X_train[i][y_train[i] == 1, 1], 'mx', label="Class 1")  # 品红色
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.legend()

N_valid = 10000
X_valid, y_valid = generate_q1_data(N_valid, pdf_params)

axes[1, 1].set_title(r"Validation $D^{%d}$" % N_valid)
axes[1, 1].plot(X_valid[y_valid == 0, 0], X_valid[y_valid == 0, 1], 'yo', label="Class 0")
axes[1, 1].plot(X_valid[y_valid == 1, 0], X_valid[y_valid == 1, 1], 'k+', label="Class 1")
axes[1, 1].set_xlabel(r"$x_1$")
axes[1, 1].set_ylabel(r"$x_2$")
axes[1, 1].legend()

x1_lim = (np.floor(X_valid[:, 0].min()), np.ceil(X_valid[:, 0].max()))
x2_lim = (np.floor(X_valid[:, 1].min()), np.ceil(X_valid[:, 1].max()))
plt.setp(axes, xlim=x1_lim, ylim=x2_lim)
plt.tight_layout()
plt.show()

def compute_discriminant_scores(X, pdf_params):
    class_0_ll = (
        pdf_params['gmm_a'][0] * mvn.pdf(X, pdf_params['mu'][0], pdf_params['Sigma'][0]) +
        pdf_params['gmm_a'][1] * mvn.pdf(X, pdf_params['mu'][1], pdf_params['Sigma'][1])
    )
    class_1_ll = mvn.pdf(X, pdf_params['mu'][2], pdf_params['Sigma'][2])
    return np.log(class_1_ll) - np.log(class_0_ll)

disc_scores_valid = compute_discriminant_scores(X_valid, pdf_params)

roc_erm, gammas_empirical = estimate_roc(disc_scores_valid, y_valid, np.bincount(y_valid))

fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
ax_roc.plot(roc_erm['p10'], roc_erm['p11'], color='purple', label="Empirical ERM Classifier ROC Curve")
ax_roc.set_xlabel(r"False Positive Rate $p(D=1 | L=0)$")
ax_roc.set_ylabel(r"True Positive Rate $p(D=1 | L=1)$")
plt.grid(True)
plt.legend()

prob_error_empirical = np.dot(
    np.array((roc_erm['p10'], 1 - roc_erm['p11'])).T, np.bincount(y_valid) / N_valid
)
min_prob_error_empirical = np.min(prob_error_empirical)
min_ind_empirical = np.argmin(prob_error_empirical)

gamma_map = pdf_params['priors'][0] / pdf_params['priors'][1]
decisions_map = disc_scores_valid >= np.log(gamma_map)
class_metrics_map = get_binary_classification_metrics(decisions_map, y_valid, np.bincount(y_valid))
min_prob_error_map = (
    class_metrics_map['FPR'] * pdf_params['priors'][0] +
    class_metrics_map['FNR'] * pdf_params['priors'][1]
)

ax_roc.plot(roc_erm['p10'][min_ind_empirical], roc_erm['p11'][min_ind_empirical], 'go',
            label="Empirical Min Pr(error)", markersize=14)
ax_roc.plot(class_metrics_map['FPR'], class_metrics_map['TPR'], 'rx',
            label="Theoretical Min Pr(error)", markersize=14)
plt.show()

print(f"Empirical: Min Pr(error) = {min_prob_error_empirical:.4f}, Min Gamma = {np.exp(gammas_empirical[min_ind_empirical]):.3f}")
print(f"Theoretical: Min Pr(error) = {min_prob_error_map:.4f}, Min Gamma = {gamma_map:.3f}")

gamma_empirical = np.exp(gammas_empirical[min_ind_empirical])
decisions = disc_scores_valid >= np.log(gamma_empirical)

correct_0 = X_valid[(decisions == 0) & (y_valid == 0)]
correct_1 = X_valid[(decisions == 1) & (y_valid == 1)]
incorrect_0 = X_valid[(decisions == 1) & (y_valid == 0)]
incorrect_1 = X_valid[(decisions == 0) & (y_valid == 1)]

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(correct_0[:, 0], correct_0[:, 1], c='cyan', label='Correctly classified 0')
ax.scatter(correct_1[:, 0], correct_1[:, 1], c='orange', label='Correctly classified 1')
ax.scatter(incorrect_0[:, 0], incorrect_0[:, 1], c='purple', label='Incorrectly classified 0')
ax.scatter(incorrect_1[:, 0], incorrect_1[:, 1], c='brown', label='Incorrectly classified 1')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Decision Boundary')
plt.legend()
plt.show()
