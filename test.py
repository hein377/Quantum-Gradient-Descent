import numpy as np
import neural_network_setup
import neural_network_instantiation
import timeit
import functools

def make_highest_one(arr):                #returns same size numpy array with 0's everywhere except 1 at the index with the max value
    max_val_index = arr.T[0].tolist().index(max(arr))
    return np.array([[1] if x==max_val_index else [0] for x in range(len(arr))])

arr = np.array([[0.94899204],
                [0.22264222],
                [0.9260522 ],
                [0.99986939],
                [0.99998753],
                [0.0033395 ],
                [0.12350486],
                [0.96521419],
                [0.19376885],
                [0.85427314]])

'''w_h1_i_vals = [1, 2, 3, 4, 5]
w_h1i_gradient_vals = [6, 7, 8, 9, 10]

cur_w_h1i_vals = [[3] for i in range(5)]
cur_w_h1i_gradient_vals = [[3] for i in range(5)]
#for i in range(5): cur_w_h1i_vals[i].append(w_h1_i_vals[i])
#for i in range(5): cur_w_h1i_gradient_vals[i].append(w_h1i_gradient_vals[i])
cur_w_h1i_vals = [cur_w_h1i_vals[i].extend([w_h1_i_vals[i]]) for i in range(5)]
cur_w_h1i_gradient_vals = [cur_w_h1i_gradient_vals[i].extend([w_h1i_gradient_vals[i]]) for i in range(5)]

print(cur_w_h1i_vals)
print(cur_w_h1i_gradient_vals)
input()'''

w_h1_i_vals = np.arange(0,30).reshape((6, 5))
w_h1i_gradient_vals = np.arange(0,30).reshape((6, 5))
cur_w_h1i_vals = [[i for i in range(10000)] for i in range(5)]
cur_w_h1i_gradient_vals = [[i for i in range(10000)] for i in range(5)]


def test1(w_h1_i_vals, w_h1i_gradient_vals, cur_w_h1i_vals, cur_w_h1i_gradient_vals):
    cur_w_h1i_vals = [cur_w_h1i_vals[i] + [w_h1_i_vals[0][i]] for i in range(5)]
    cur_w_h1i_gradient_vals = [cur_w_h1i_gradient_vals[i] + [w_h1i_gradient_vals[0][i]] for i in range(5)]
    return cur_w_h1i_vals, cur_w_h1i_gradient_vals

def test2(w_h1_i_vals, w_h1i_gradient_vals, cur_w_h1i_vals, cur_w_h1i_gradient_vals):
    for i in range(5): cur_w_h1i_vals[i].append(w_h1_i_vals[0][i])
    for i in range(5): cur_w_h1i_gradient_vals[i].append(w_h1i_gradient_vals[0][i])
    return cur_w_h1i_vals, cur_w_h1i_gradient_vals

t = timeit.Timer(functools.partial(test1, w_h1_i_vals, w_h1i_gradient_vals, cur_w_h1i_vals, cur_w_h1i_gradient_vals)) 
print(t.timeit(1000))
t = timeit.Timer(functools.partial(test2, w_h1_i_vals, w_h1i_gradient_vals, cur_w_h1i_vals, cur_w_h1i_gradient_vals)) 
print(t.timeit(1000))

'''print(timeit.repeat(stmt=test1(w_h1_i_vals, w_h1i_gradient_vals, cur_w_h1i_vals, cur_w_h1i_gradient_vals), number=100,repeat=2))
print(timeit.repeat(stmt=test2(w_h1_i_vals, w_h1i_gradient_vals, cur_w_h1i_vals, cur_w_h1i_gradient_vals), number=100,repeat=2))'''