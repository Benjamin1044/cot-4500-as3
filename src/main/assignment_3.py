import numpy as np


def function(t, w):
  return t - (w**2)


def euler_method(t, w, num_of_iterations, max_range):
  h_1 = (max_range - t) / num_of_iterations

  for cur_iteration in range(0, num_of_iterations):
    w = w + (h_1 * function(t, w))
    t = t + h_1

  print("%.5f" % w)


def runge_kutta(t, w, num_of_iterations, max_range):
  h_2 = (max_range - t) / num_of_iterations

  for cur_iteration in range(0, num_of_iterations):
    k1 = h_2 * function(t, w)
    k2 = h_2 * function(t + h_2 / 2, w + k1 / 2)
    k3 = h_2 * function(t + h_2 / 2, w + k2 / 2)
    k4 = h_2 * function(t + h_2, w + k3)

    t = t + h_2
    w = w + (k1 + 2 * k2 + 2 * k3 + k4) / 6

  print("%.5f" % w)


def gauss_jordan(A):
  n = A.shape[0]

  for i in range(n):
    pivot_p = i
    if (A[pivot_p, i]) == 0:
      pivot_p = pivot_p + 1

    A[[i, pivot_p]] = A[[pivot_p, i]]

    for j in range(i + 1, n):
      factor = A[j, i] / A[i, i]
      A[j, i:] = A[j, i:] - factor * A[i, i:]

  x = np.zeros(n)
  for i in range(n - 1, -1, -1):
    x[i] = (A[i, -1] - np.dot(A[i, i:-1], x[i:])) / A[i, i]

  return (x)


def factorization(B):
  n = B.shape[0]
  L_matrix = np.zeros((n, n))
  U_matrix = np.zeros((n, n))
  determinant = np.linalg.det(B)
  for i in range(n):
    L_matrix[i, i] = 1
    for j in range(i , n):
      U_matrix[i, j] = (B[i, j] - np.dot(L_matrix[i, :i], U_matrix[:i, j]))
    for k in range(i + 1 , n):
      L_matrix[k, i] = (B[k, i] - np.dot(L_matrix[k, :i], U_matrix[:i, i])) / U_matrix[i, i]

  print("%.5f" % determinant)
  print("")
  print(L_matrix, "\n")
  print(U_matrix, "\n")


def diagonally_dominant(C):
  n = len(C)
  total = 0
  for i in range(0, n):
    for j in range(0, n):
      if j != i:
        total = total + abs(C[i][j])
  
  if abs(C[i][i]) < total:
        print("False\n")
  else:
        print("True\n")  

def positive_definite(D):
    n = len(D)  
    for i in range(n):
      for j in range(n):
          if (D[i][j] != D[j][i]):
            print("False\n")
            break
  
    values = np.linalg.eigvals(D)

    if np.all(values > 0):
      print("True\n")
    else:
        print("False\n")
  



if __name__ == "__main__":
  t = 0
  w = 1
  max_range = 2
  num_of_iterations = 10
  euler_method(t, w, num_of_iterations, max_range)
  print("")
  runge_kutta(t, w, num_of_iterations, max_range)
  print("")
  A = np.array([[2, -1, 1, 6], [1, 3, 1, 0], [-1, 5, 4, -3]])
  x = gauss_jordan(A)
  print(x)
  print("")
  B = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]], dtype = np.double)
  factorization(B)
  C = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 2, 3], [4, 2, 3, 12, 2], [3, 2, 4, 0, 8]], dtype = np.double)
  diagonally_dominant(C)
  D = np.array([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
  positive_definite(D)
  