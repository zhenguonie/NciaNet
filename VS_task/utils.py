import numpy as np
import paddle
from math import sqrt
from sklearn.linear_model import LinearRegression

def cos_formula(a, b, c):
    ''' formula to calculate the angle between two edges
        a and b are the edge lengths, c is the angle length.
    '''
    res = (a**2 + b**2 - c**2) / (2 * a * b)
    # sanity check
    res = -1. if res < -1. else res
    res = 1. if res > 1. else res
    return np.arccos(res)

def setxor(a, b):
    n = len(a)
    
    res = []
    link = []
    i, j = 0, 0
    while i < n and j < n:
        if a[i] == b[j]:
            link.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            res.append(a[i])
            i += 1
        else:
            res.append(b[j])
            j += 1
    
    if i < j:
        res.append(a[-1])
    elif i > j:
        res.append(b[-1])
    else:
        link.append(a[-1])
    
    return res, link

def calculate_ef_1percent(y_true_list_all, y_hat_list_all):
    if len(y_hat_list_all) == 0 or len(y_true_list_all) == 0:
        raise ValueError("y_hat_list_all or y_true_list_all empty, cannot calculate EF 1%")
    y_hat_all = np.array(y_hat_list_all)
    y_true_all = np.array(y_true_list_all)
    total_samples = len(y_hat_all)
    top_1_percent_count = max(1, int(total_samples * 0.01))
    paired_data = list(zip(y_hat_all, y_true_all))
    paired_data_sorted = sorted(paired_data, key=lambda x: x[0], reverse=True)
    sorted_y_hat, sorted_y_true = zip(*paired_data_sorted)
    top_1_percent_samples = list(zip(sorted_y_hat[:top_1_percent_count], sorted_y_true[:top_1_percent_count]))
    predicted_and_real_active = sum(1 for pred, real in top_1_percent_samples if pred >= 0.5 and real == 1)
    total_active_count = np.sum(y_true_all == 1)  
    ef_1_percent = predicted_and_real_active / total_active_count
    return ef_1_percent


def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def mae(y,f):
    mae = (np.abs(y-f)).mean()
    return mae

def sd(y,f):
    f,y = f.reshape(-1,1),y.reshape(-1,1)
    lr = LinearRegression()
    lr.fit(f,y)
    y_ = lr.predict(f)
    sd = (((y - y_) ** 2).sum() / (len(y) - 1)) ** 0.5
    return sd

def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp

def generate_segment_id(index):
    zeros = paddle.zeros(index[-1] + 1, dtype="int32")
    index = index[:-1]
    segments = paddle.scatter(
        zeros, index, paddle.ones_like(
                index, dtype="int32"), overwrite=False)
    segments = paddle.cumsum(segments)[:-1] - 1
    return segments