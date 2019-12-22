import numpy as np
import matplotlib.pyplot as plt
import pickle

# Plot simple sinus function
fig_handle = plt.figure()
x = np.linspace(0, 2 * np.pi)
y = np.sin(x)
plt.plot(x, y)
plt.show()
# Save figure handle to disk
pickle.dump(fig_handle, open('sinus.pickle', 'wb'))
figx = pickle.load(open('sinus.pickle', 'rb'))
figx.show()
