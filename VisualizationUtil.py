import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def show_transformation_2d(matrix, grid_size=11, show_base=True, size=5):
    x_vals = np.linspace(-1, 1, grid_size)
    y_vals = np.linspace(-1, 1, grid_size)

    if show_base:
        fig, ax = plt.subplots(figsize=(size, size))

        # Plot original grid (blue lines)
        for x in x_vals:
            original_line = np.array([[x, x], [-1, 1]])
            ax.plot(original_line[0], original_line[1], color='blue', alpha=0.5)

        for y in y_vals:
            original_line = np.array([[-1, 1], [y, y]])
            ax.plot(original_line[0], original_line[1], color='blue', alpha=0.5)

        # Highlight original axes (x=0 and y=0) in black
        ax.plot([0, 0], [-1, 1], color='black', linewidth=2)
        ax.plot([-1, 1], [0, 0], color='black', linewidth=2)

        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_title('Original 2D Grid (Before Transformation)')
        plt.grid(True)
        plt.show()

    fig, ax = plt.subplots(figsize=(size, size))

    # Plot transformed grid (red lines)
    for x in x_vals:
        transformed_line = matrix @ np.array([[x, x], [-1, 1]])
        ax.plot(transformed_line[0], transformed_line[1], color='red', alpha=0.5)

    for y in y_vals:
        transformed_line = matrix @ np.array([[-1, 1], [y, y]])
        ax.plot(transformed_line[0], transformed_line[1], color='red', alpha=0.5)

    # Highlight transformed axes (x=0 and y=0) in black
    transformed_axes = matrix @ np.array([
        [[0, 0], [-1, 1]],  # X axis (transformed)
        [[-1, 1], [0, 0]]   # Y axis (transformed)
    ])

    ax.plot(transformed_axes[0, 0], transformed_axes[1, 0], color='black', linewidth=2)
    ax.plot(transformed_axes[0, 1], transformed_axes[1, 1], color='black', linewidth=2)

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_title('Transformed 2D Grid')
    plt.grid(True)
    plt.show()



def show_transform_3d(matrix, grid_size=5, show_base=True, size=5):
    x_vals = np.linspace(-1, 1, grid_size)
    y_vals = np.linspace(-1, 1, grid_size)
    z_vals = np.linspace(-1, 1, grid_size)

    if show_base:
        fig = plt.figure(figsize=(size, size))
        ax = fig.add_subplot(111, projection='3d')

        # Plot original grid (blue lines) along all three axes
        for x in x_vals:
            for y in y_vals:
                original_line = np.array([[x, x], [y, y], [-1, 1]])
                ax.plot(original_line[0], original_line[1], original_line[2], color='blue', alpha=0.5)

        for y in y_vals:
            for z in z_vals:
                original_line = np.array([[-1, 1], [y, y], [z, z]])
                ax.plot(original_line[0], original_line[1], original_line[2], color='blue', alpha=0.5)

        for x in x_vals:
            for z in z_vals:
                original_line = np.array([[x, x], [-1, 1], [z, z]])
                ax.plot(original_line[0], original_line[1], original_line[2], color='blue', alpha=0.5)

        # Highlight original axes (x=0, y=0, z=0) in black
        ax.plot([0, 0], [-1, 1], [0, 0], color='black', linewidth=2)
        ax.plot([0, 0], [0, 0], [-1, 1], color='black', linewidth=2)
        ax.plot([-1, 1], [0, 0], [0, 0], color='black', linewidth=2)

        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('Original 3D Grid (Before Transformation)')
        plt.show()

    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection='3d')

    # Plot transformed grid (red lines) along all three axes
    for x in x_vals:
        for y in y_vals:
            transformed_line = matrix @ np.array([[x, x], [y, y], [-1, 1]])
            ax.plot(transformed_line[0], transformed_line[1], transformed_line[2], color='red', alpha=0.5)

    for y in y_vals:
        for z in z_vals:
            transformed_line = matrix @ np.array([[-1, 1], [y, y], [z, z]])
            ax.plot(transformed_line[0], transformed_line[1], transformed_line[2], color='red', alpha=0.5)

    for x in x_vals:
        for z in z_vals:
            transformed_line = matrix @ np.array([[x, x], [-1, 1], [z, z]])
            ax.plot(transformed_line[0], transformed_line[1], transformed_line[2], color='red', alpha=0.5)

    # Highlight transformed axes (x=0, y=0, z=0) in black
    transformed_axes = matrix @ np.array([
        [[0, 0], [0, 0], [-1, 1]],  # X axis (transformed)
        [[0, 0], [-1, 1], [0, 0]],  # Y axis (transformed)
        [[-1, 1], [0, 0], [0, 0]]   # Z axis (transformed)
    ])

    ax.plot(transformed_axes[0, 0], transformed_axes[1, 0], transformed_axes[2, 0], color='black', linewidth=2)
    ax.plot(transformed_axes[0, 1], transformed_axes[1, 1], transformed_axes[2, 1], color='black', linewidth=2)
    ax.plot(transformed_axes[0, 2], transformed_axes[1, 2], transformed_axes[2, 2], color='black', linewidth=2)

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('Transformed 3D Grid')
    plt.show()
