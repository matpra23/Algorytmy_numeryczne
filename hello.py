import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 70, 200)
plt.plot(x, np.tan(x))
plt.show()