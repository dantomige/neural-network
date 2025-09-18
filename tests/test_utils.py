from nn.utils import matrix_multiply, dot_product, DimensionError
import pytest

class TestMatrixMultiply:

    def test_matrices():
        A = [           # 2 X 2
            [1, 0],
            [0, 1]
        ]

        B = [           # 2 X 3
            [1, 0, 1],
            [1, 0.5, 0]
        ]

        C = [           # 3 X 4
            [0.5, 0, 0, 1],
            [2, 0, -2, 0],
            [1, 1, 4, -1],
        ]

        D = [           # 4 X 1
            [1],
            [3],
            [3],
            [3]
        ]

        AB_expected = B  # 2 X 3

        BC_expected = [  # 2 X 4
            [1.5, 1, 4, 0],
            [1.5, 0, -1, 1],
        ]
        CD_expected = [  # 3 X 1
            [3.5],
            [0],
            [13],
        ]


        assert matrix_multiply(A, B) == AB_expected
        assert matrix_multiply(B, C) == BC_expected
        assert matrix_multiply(C, D) == CD_expected
        
        with pytest.raises(DimensionError):
            matrix_multiply(A, C)

        with pytest.raises(DimensionError):
            matrix_multiply(A, D)

        with pytest.raises(DimensionError):
            matrix_multiply(B, D)
    
        

class TestDotProduct():

    def test_dot():
        a = [1, 0, 1]
        b = [4, 1]
        c = [3, -1, 1]
        d = [1, 0, -1]

        ac = 4
        cd = 2
        da = 0

        assert dot_product(a, c) == ac
        assert dot_product(d, c) == cd
        assert dot_product(a, d) == da

        with pytest.raises(DimensionError):
            dot_product(a, b)

        with pytest.raises(DimensionError):
            dot_product(b, a)

        with pytest.raises(DimensionError):
            dot_product(c, d)