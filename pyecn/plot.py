import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("profiles\hppc_pulse.csv", header=None)

# Plot
plt.figure()
plt.plot(df.iloc[:, 0], df.iloc[:, 1])
plt.xlabel("Column 1")
plt.ylabel("Column 2")
plt.title("2-Column Graph")
plt.show()
