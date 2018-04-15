
import numpy as np

vector=np.array(range(0,9))
print(vector)

matrix=np.reshape(vector,[3, 3])
print(matrix)

Q=np.load('q.npy')
print(Q[:40,:5])
print(np.amax(Q))