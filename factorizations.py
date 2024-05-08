# Author: Wren Taylor

import numpy as np  
import matplotlib.pyplot as plt
import numpy.linalg as la
import scipy.linalg as scila
import time 

def householder_reflection(a):
    """Create the Householder matrix that will zero out the sub-diagonal elements of matrix a."""
    v = a.copy()
    v[0] += np.sign(a[0]) * np.linalg.norm(a)
    v = v / np.linalg.norm(v)
    H = np.eye(len(a)) - 2 * np.outer(v, v)
    return H

def qr_pivoting_householder(A):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    P = np.eye(n)

    for i in range(min(m, n)):
        # Pivoting based on the maximum norm of remaining columns
        max_col = np.argmax(np.linalg.norm(R[:, i:], axis=0)) + i
        P[:, [i, max_col]] = P[:, [max_col, i]]
        R[:, [i, max_col]] = R[:, [max_col, i]]

        # Apply Householder reflection
        H = np.eye(m)
        H[i:, i:] = householder_reflection(R[i:, i])
        R = np.dot(H, R)
        Q = np.dot(Q, H.T)

    return Q, R, P

#These subroutines are from Lab 12 
def driver():
    ''' Create an ill-conditioned rectangular matrix '''
    N = 10
    M = 5
    A = create_rect(N, M)     
    b = np.random.rand(N, 1)

    Q, R, P, rank = rrqr_householder(A)
    print("Rank of A:", rank) 
    x = solve_linear_system(A, b)
    print("Solution x:", x)
    

def create_rect(N,M):
     ''' this subroutine creates an ill-conditioned rectangular matrix'''
     a = np.linspace(1,10,M)
     d = 10**(-a)
     
     D2 = np.zeros((N,M))
     for j in range(0,M):
        D2[j,j] = d[j]
     
     '''' create matrices needed to manufacture the low rank matrix'''
     A = np.random.rand(N,N)
     print('A:', A)

     Q1, R = la.qr(A)
     print('Q1:', Q1)
    
     test1 = np.matmul(Q1,R)
     print('test1:', test1)

     A = np.random.rand(M,M)
     print('A:', A)

     Q2,R = la.qr(A)
     print('Q2:', Q2)

     test2 = np.matmul(Q2,R)
     print('test2:', test2)
     
     B = np.matmul(Q1,D2)
     B = np.matmul(B,Q2)
     print('B:', B)
     return B 
     

if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()



