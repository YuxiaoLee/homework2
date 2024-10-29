import matplotlib.pyplot as plt
import numpy as np

xy_true = (0.25, 0.25)

sensors = [
    (1 / np.sqrt(2), 1 / np.sqrt(2)),
    (-1 / np.sqrt(2), 1 / np.sqrt(2)),
    (-1 / np.sqrt(2), -1 / np.sqrt(2)),
    (1 / np.sqrt(2), -1 / np.sqrt(2))
]

# Standard deviations
sigma_x, sigma_y, sigma_i = 0.25, 0.25, 0.01


# Function to calculate Euclidean distance between two points
def dist_points(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Function to generate a valid sensor measurement
def measure_value(sensor_pos):
    while True:
        measurement = dist_points(sensor_pos, xy_true) + np.random.normal(scale=sigma_i)
        if measurement > 0:
            return measurement


# Collect measurements from each sensor
sensor_measurements = {s: measure_value(s) for s in sensors}

# Display sensor data
print("Sensor data:")
for i, (sensor, measurement) in enumerate(sensor_measurements.items(), 1):
    print(f"Distance of Sensor {i} from true position: {measurement:.3f}")

# Contour levels and meshgrid for plotting
contour_level = np.arange(0, 300, 10)
x = np.linspace(-2, 2, 1000)
y = np.linspace(-2, 2, 1000)
X, Y = np.meshgrid(x, y)


# Objective functions based on the number of sensors used
def f0(x, y):
    return (x ** 2 / sigma_x ** 2) + (y ** 2 / sigma_y ** 2)


def f1(x, y):
    return f0(x, y) + (sensor_measurements[sensors[0]] - dist_points(sensors[0], (x, y))) ** 2 / sigma_i ** 2


def f2(x, y):
    return f1(x, y) + (sensor_measurements[sensors[2]] - dist_points(sensors[2], (x, y))) ** 2 / sigma_i ** 2


def f3(x, y):
    return f2(x, y) + (sensor_measurements[sensors[1]] - dist_points(sensors[1], (x, y))) ** 2 / sigma_i ** 2


def f4(x, y):
    return f3(x, y) + (sensor_measurements[sensors[3]] - dist_points(sensors[3], (x, y))) ** 2 / sigma_i ** 2


# Plotting function for each objective function
def plot_function(f, num_sensors):
    Z = f(X, Y)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.contourf(X, Y, Z, levels=contour_level, cmap='coolwarm')
    ax.plot(*xy_true, '+', markersize=20, color='r', label='True Position', mew=2)

    # Plot the sensors used
    for i, s in enumerate(sensors[:num_sensors]):
        ax.plot(*s, 'o', markersize=10, color='b', label=f'Sensor {i + 1}')

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.legend()
    return fig


# Generate plots for each objective function
figures = [plot_function(f, k) for f, k in zip([f0, f1, f2, f3, f4], range(5))]

# Add titles and labels
for i, fig in enumerate(figures):
    fig.suptitle(f'MAP Objective Function Contours for K={i}', fontsize=20)
    fig.axes[0].set_xlabel('X', fontsize=16)
    fig.axes[0].set_ylabel('Y', fontsize=16)

# Display the plots
plt.show()
