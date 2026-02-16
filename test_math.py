from engine import fit_yield_curve
import numpy as np

mats = np.array([1, 2, 5, 10, 30])
yields = np.array([0.04, 0.042, 0.045, 0.048, 0.05])
params = fit_yield_curve(mats, yields)
print(f"Success! Fitted Parameters: {params}")