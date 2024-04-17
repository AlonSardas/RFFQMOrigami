# RFFQMOrigami

Utility for generating and folding *rigidly and flat-foldable quadrilateral mesh (RFFQM) origami*.

| <img src="./img/cylinder-folded.png" width="300" /> | <img src="./img/radial-example-folded.png" width="300" /> |
|-----------------------------------------------------|-----------------------------------------------------------|
| <img src="./img/spherical-cap.png" width="300" />   | <img src="./img/saddle.png" width="300" />                |

<img src="./img/classical-miura-ori.png" />

## Installation

Download the package and run
```bash
python setup.py
```

## Usage

### Creating patterns by providing boundary data
The most common usage is to create an RFFQM crease pattern 
by dictating the boundary input data and solving the marching algorithm.
Then, the pattern can be rigidly folded.

```python
import numpy as np

from origami import marchingalgorithm, quadranglearray, RFFQMOrigami, origamiplots

angle = 0.7 * np.pi  # The base angle of the Miura-Ori
ls = np.ones(10)  # The lengths for the left boundary vertical creases
cs = np.ones(10)  # The lengths for the bottom boundary horizontal creases

# Create the boundary angles for classical Miura-Ori
angles_left, angles_bottom = marchingalgorithm.create_miura_angles(ls, cs, angle)

# Here we can add perturbations to the lengths and to the angles by setting
# ls, cs, angles_left, angles_bottom

# Here the marching algorithm is applied to calculate the angles along the crease pattern
marching = marchingalgorithm.MarchingAlgorithm(angles_left, angles_bottom)
# Here the marching algorithm is applied to create the crease pattern based on the given boundary lengths
dots, indexes = marching.create_dots(ls, cs)

# QuadrangleArray is a data structure that handles quadrangle mesh objects
quads = quadranglearray.dots_to_quadrangles(dots, indexes)
ori = RFFQMOrigami.RFFQM(quads)  # The origami object allows the folding of the given flat crease pattern

# Plot with an interactive slider for the folding angle 
origamiplots.plot_interactive(ori)
```

### Creating patterns by providing principal curvatures
We use a specific perturbation class for the angles and lengths, called *Alternating angles*,
using which we can dictate the desired principal curvatures.
`kx(x)` and `ky(y)`. These are the principal curvatures in `x` and `y` directions respectively.
They both should be defined on the segment `[0,1]`.

```python
import numpy as np
from matplotlib import pyplot as plt

from origami.alternatingpert import curvatures, utils

kx = lambda x: 0.3 * np.sin(2 * np.pi * x)
ky = lambda y: 0.6 * (y - 0.5)

W0 = 2.3  # the target activation angle for the origami
theta = 1.2  # the base Miura-Ori angle

# initial values for the angles and lengths perturbation
delta0, Delta0 = -0.3, 0.5

Nx, Ny = 12, 10  # number of unit cells to use
L_tot, C_tot = 15, 15  # Roughly the size of the flat pattern

# Find smooth functions for the perturbation according to the desired curvatures
delta_func = curvatures.get_delta_func_for_kx(L_tot, C_tot, W0, theta, kx, delta0)
Delta_func = curvatures.get_Delta_func_for_ky(L_tot, C_tot, W0, theta, ky, Delta0)

# Plot the perturbations
fig, axes = plt.subplots(2)
utils.plot_perturbations(axes, delta_func, Delta_func, Nx, Ny)
plt.show()

# Build the pattern based on the perturbations
ori = utils.create_perturbed_origami(theta, Ny, Nx, L_tot, C_tot, delta_func, Delta_func)

# Fold to the target activation angle
ori.set_gamma(ori.calc_gamma_by_omega(W0))
```

### Ploting the patterns
Given RFFQM origami object we can plot it
1) in a static configuration:
   ```python
   from matplotlib import pyplot as plt
   
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   
   # ori object is created above
   ori.dots.plot(ax, panel_color='C1', alpha=1.0,
             edge_alpha=1.0, edge_color='g')
   ax.set_aspect('equal')
   plt.show()
   ```
2) with a slider for the activation angle:
   ```python
   from origami import origamiplots
   
   origamiplots.plot_interactive(ori)
   ```

### Calculating the discrete geometric properties
Given an origami pattern, we can calculate its geometric properties.
```python
from origami import origamimetric

geometry = origamimetric.OrigamiGeometry(ori.dots)
# Calculate the metric entries for each cell
geometry.get_metric()
# Calculate the second fundamental from entries for each cell
geometry.get_SFF()
# Calculate the Gaussian and mean curvatures
Ks, Hs = geometry.get_curvatures_by_shape_operator()
```
