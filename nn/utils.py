class DimensionError(Exception):
    pass

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

def element_multiply(A: list[list[int]], B: list[list[int]]):
    dims_A, dims_B = dims(A), dims(B)

    assert dims_A[0] == 1
    assert dims_A[1] == dims_B[0]

    output = []

    for col in zip(*A):
        for row in B:
            mag = col[0]
            output.append(scale(mag, row))
    return output

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