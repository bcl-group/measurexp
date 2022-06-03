from notes.notes._nmf import non_negative_factorization
import cupy as cp
import numpy as np

from sklearn import decomposition
X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
W, H, n_iter = decomposition.non_negative_factorization(X, n_components=2, init='random', random_state=0)
print(W, H, n_iter)

X = cp.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
W, H, n_iter = non_negative_factorization(X, n_components=2, init='random', random_state=0)
print(W, H, n_iter)

# print(str(type(X)))
# print(X.dtype)
# cp.astype()

# W, H, n_iter = non_negative_factorization(X, n_components=2,
#     init='random', random_state=0)
