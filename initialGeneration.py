import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# Configuration
num_particles = 8
sigma = 0.15
# Compute the mask based on distance from the center
particles = []

# Set up the figure
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xticks(np.arange(-2, 2.1, 0.5))
ax.set_yticks(np.arange(-2, 2.1, 0.5))
ax.grid(True)
ax.set_title(f"Click to place {num_particles} particles")

circle_x = -0.2
circle_y = 0.0
circle_radius = 0.0

square_center_x = 0.8
square_center_y = 1.2
square_half_size = 0.0
from matplotlib.patches import Circle, Rectangle

# --- circle outline ---
ax.add_patch(
    Circle((circle_x, circle_y), circle_radius,
           fill=False, edgecolor="red", linewidth=2)
)

# --- square outline ---
ax.add_patch(
    Rectangle((square_center_x - square_half_size, square_center_y - square_half_size),
              2*square_half_size, 2*square_half_size,
              fill=False, edgecolor="red", linewidth=2)
)



# Callback function for mouse clicks
def on_click(event):
    global particles
    if event.inaxes is not None and len(particles) < num_particles:
        x, y = event.xdata, event.ydata

        # Check if inside invariance area
        if (x - circle_x) ** 2 + (y - circle_y) ** 2 <= circle_radius ** 2:
            print("Invalid placement: Inside invariance area!")
            return

        particles.append({"position": [x, y], "sigma": sigma})
        ax.plot(x, y, 'bo')  # Mark point in blue
        plt.draw()

        if len(particles) == num_particles:
            save_particles()


# Function to save particles to a JSON file
def save_particles():
    with open("initial_particles8.json", "w") as file:
        json.dump(particles, file)
    print("Particle positions saved to 'initial_particles.json'.")
    plt.close()


# Connect the event listener
fig.canvas.mpl_connect("button_press_event", on_click)

# Show the interactive plot
plt.show()
