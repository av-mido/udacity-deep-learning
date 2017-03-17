"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np
import math
import sys

print(sys.version_info)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def softmax_long_version(x):
    """Compute softmax values for each sets of scores in x."""
    if isinstance(x, list):
        denom = sum([math.exp(j) for j in x])
        return [math.exp(j)/denom for j in x]
    elif x.ndim == 1:
        denom = sum([math.exp(j) for j in x])
        return np.array([math.exp(j)/denom for j in x], copy=True)
    else:
        result = x.copy()
        # print('num cols: ', x.shape[1])
        for colidx in range(0,x.shape[1]):
            result[:,colidx] = softmax(x[:,colidx])
        return result

if __name__ == "__main__":
    # Plot softmax curves
    import matplotlib.pyplot as plt
    x = np.arange(-2.0, 6.0, 0.1)
    scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
    print('scores: ', scores)
    softmax_scores = softmax(scores)
    plt.plot(x, softmax_scores.T, linewidth=2)
    print('softmax scores: ', softmax_scores)
    plt.show()