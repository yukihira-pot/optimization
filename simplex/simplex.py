import numpy as np

class SimplexSolver:
    """
    単体法を用いて線型計画問題を解く

    minimize c^T x
    subject to Ax = b

    x, c in R^n
    A in R^(m*n)
    b in R^m

    Attributes
    --------
    determine_basic_vars : set
        初期の基底変数のインデックスを求める

    is_optimized : bool
        最適化されているかどうか (目的関数の係数行列 c >= 0 かどうか)
        を判断

    nxt_basic_var : int
        次の基底変数のインデックスを求める
    
    nxt_nonbasic_var : int
        次の非基底変数のインデックスを求める
    
    renew_basic_vars : None
        基底変数の集合を更新
    
    reduce_row : None
        新しい基底変数について掃き出し

    """
    def __init__(self, A: np.array, b: np.array, c: np.array) -> None:
        self.A, self.b, self.c = A, b, c
        self.m, self.n = A.shape
        assert b.shape == (self.m, ), "shape of b invalid"
        assert c.shape == (self.n, ), "shape of c invalid"
        self.basic_vars = set()

    def determine_basic_vars(self) -> set:
        """
        Parameters
        -------
        A : np.array
            制約条件の係数行列

        Returns
        --------
        self.basic_vars : set
            基底変数のインデックスの集合
        """
        A_tmp = np.empty((self.m, 0))

        for i in range(self.n): # ここどうにかしたい
            try:
                rank_A_tmp = np.linalg.matrix_rank(A_tmp)
            except:
                rank_A_tmp = 0
            insert_rank_A_tmp = np.linalg.matrix_rank(
                np.insert(A_tmp, 0, self.A[:, i], axis=1)
            )
            if insert_rank_A_tmp > rank_A_tmp:
                A_tmp = np.insert(A_tmp, 0, self.A[:, i], axis=1)
                self.basic_vars.add(i)

        return self.basic_vars


if __name__ == "__main__":
    A = np.array([[5, 2, 1, 0, 0], [1, 2, 0, 1, 0], [5, -4, 0, 0, 1]])
    b = np.array([30, 14, 15])
    c = np.array([-5, -4, 0, 0, 0])
    simplexsolver = SimplexSolver(A, b, c)
    print(simplexsolver.determine_basic_vars())
