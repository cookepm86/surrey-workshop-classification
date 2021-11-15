from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use("/Users/miccoo/Desktop/kindred.mplstyle")
kindred_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
sns.set_palette(sns.color_palette(kindred_colors))

# create blue colormap
N = 256
color1 = np.ones((N, 4))
color1[:, 0] = np.linspace(5/256, 1, N) #
color1[:, 1] = np.linspace(164/256, 1, N) #
color1[:, 2] = np.linspace(224/256, 1, N) #
color1_cmp = ListedColormap(color1)

# pink colormap
color2 = np.ones((N, 4))
color2[:, 0] = np.linspace(231/256, 1, N)
color2[:, 1] = np.linspace(59/256, 1, N)
color2[:, 2] = np.linspace(138/256, 1, N)
color2_cmp = ListedColormap(color2)

newcolors2 = np.vstack((color2_cmp(np.linspace(0, 1, 128)),
color1_cmp(np.linspace(1, 0, 128))))

double = ListedColormap(newcolors2, name='double')