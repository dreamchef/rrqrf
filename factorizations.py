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
        max_col = np.argmax(np.linalg.norm(R[i:, i:], axis=0)) + i
        P[:, [i, max_col]] = P[:, [max_col, i]]
        R[:, [i, max_col]] = R[:, [max_col, i]]

        # Apply Householder reflection
        H = np.eye(m)
        H[i:, i:] = householder_reflection(R[i:, i])
        R = np.dot(H, R)
        Q = np.dot(Q, H.T)

    return Q, R, P

def solve_linear_system(A, b):
    Q, R, P, rank = rrqr_householder(A)  
    Qb = np.dot(Q.T, b)  # Transform b by Q transpose
    
    # Solve Rx = Q^Tb using back substitution
    x = np.linalg.solve(R[:R.shape[1]], Qb[:R.shape[1]])
    return x

# QR factorization of A using Householder reflections
def rrqr_householder(A):
    m, n = A.shape
    Q = np.eye(m) 
    R = A.copy()  
    P = np.arange(n)  
    rank = 0  
    
    for j in range(n):
        # Find the column with the maximum norm from j to n
        col_norms = np.linalg.norm(R[:, j:], axis=0)
        max_col_idx = np.argmax(col_norms) + j
        
        # Swap the current column with the maximum norm column
        R[:, [j, max_col_idx]] = R[:, [max_col_idx, j]]
        Q[:, [j, max_col_idx]] = Q[:, [max_col_idx, j]]
        P[j], P[max_col_idx] = P[max_col_idx], P[j]
        
        # Apply Householder transformation to eliminate subdiagonal elements
        x = R[j:, j]
        v = np.zeros_like(x)
        v[0] = np.linalg.norm(x)
        v += np.sign(x[0]) * np.linalg.norm(x) * np.eye(len(x))[0]
        R[j:, :] -= 2 * np.outer(v, v) @ R[j:, :]
        Q[:, j:] -= 2 * Q[:, j:] @ np.outer(v, v)
        
        # Increment rank if the diagonal element is nonzero
        if np.abs(R[j, j]) > 1e-10:
            rank += 1
        
    return Q, R, P, rank


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



