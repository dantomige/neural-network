class DimensionError(Exception):
    pass

class Matrix:
    pass

class Vector:
    pass

def create_vector(values):
    return [[v] for v in values]

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def scale_matrix(mag, A):
    return [[mag * val for val in row] for row in A]

def matrix_add(A, B):
    output = []
    for rowA, rowB in zip(A, B):
        new_row = []
        for valA, valB in zip(rowA, rowB):
            new_row.append(valA + valB)
        output.append(new_row)
    return output

def matrix_multiply(A: list[list[int]], B: list[list[int]]):
    if len(A[0]) != len(B):
        raise DimensionError(f"Matrices of dimension {(len(A), len(A[0]))} and {(len(B), len(B[0]))} cannot be multiplied.")

    C = []
    for row in A:
        new_row = []
        for col in zip(*B):
            new_row.append(dot_product(row, col))
        C.append(new_row)
    return C

def element_multiply_matrix(A: list[list[int]], B: list[list[int]]):
    dims_A, dims_B = dims(A), dims(B)

    assert dims_B[1] == 1
    assert dims_A[1] == dims_B[0]

    output = []
    for col in zip(*A):
        for row in B:
            mag = row[0]
            output.append(scale(mag, col))
    return transpose(output)

def apply_func_matrix(func, A):
    return [[func(val) for val in row] for row in A]

def element_multiply_vector(a, b):
    dims_a, dims_b = dims(a), dims(b)

    assert dims_a[1] == 1
    assert dims_b[1] == 1
    output = []

    for rowa, rowb in zip(a, b):
        output.append(rowa[0] * rowb[0])

    return create_vector(output)

def dot_product(a, b):
    if len(a) != len(b):
        raise DimensionError(f"Can't find dot product of vectors of different lengths. Vectors {len(a)}, {len(b)}")
    
    output = sum(x * y for x, y in zip(a, b))

    return output

def scale(mag, a):
    return [mag * val for val in a]

def identity(dim):
    output = [([0] * dim) for _ in range(dim)]

    for i in range(dim):
        output[i][i] = 1

    return output

def dims(A):
    return (len(A), len(A[0]))

if __name__ == "__main__":
    pass