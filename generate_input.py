import numpy as np

# Dimensions matching the short_test PBS script
x_ext = 50
y_ext = 50
z_ext = 50

# Material IDs from compiler_opts.h
# Ensure these match the active enum in the C++ code
AIR = 1
PORE = 2
ROCK = 3
SULPHIDE = 4

# Generate Domain
# Numpy shape is (slowest, ..., fastest). 
# Based on main.cpp: i=zext, j=yext, k=xext. 
# ZFastest scheme implies k (xext) is the fastest varying index.
shape = (z_ext, y_ext, x_ext)

# Initialise mostly Rock
data = np.full(shape, ROCK, dtype=np.uint8)

# 1. Create a central Pore channel (Vertical flow along Z)
#    We make a 10x10 channel running through the center
data[:, 20:30, 20:30] = PORE

# 2. Place a Sulphide block inside the channel
#    This ensures the sulphide is in contact with the pore for reaction
data[20:30, 22:28, 22:28] = SULPHIDE

# 3. Add Air layers at top and bottom to drive pressure/flow
#    (Assuming Z-axis is the primary flow direction)
data[0:2, :, :] = AIR   # "Inlet"
data[-2:, :, :] = AIR   # "Outlet"

# Write to File
filename = "synthetic_test_input.raw"
data.tofile(filename)

print(f"Generated {filename}")
print(f"Dimensions: {x_ext} x {y_ext} x {z_ext}")
print(f"Total size: {data.size} bytes")