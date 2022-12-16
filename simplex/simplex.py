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
        self.basis_matrix = np.empty((self.m, 0))
        self.non_basis_matrix = np.empty((self.m, 0))
        assert np.linalg.matrix_rank(self.A) == self.m, "rank of A invalid"
        assert b.shape == (self.m, ), "shape of b invalid"
        assert c.shape == (self.n, ), "shape of c invalid"
        # 基底変数が最初の m 個になるように、determine_basic_vars() で設定する
        self.basic_vars = [i for i in range(self.m)]

    def determine_basic_vars(self) -> None:
        """
        Notes
        --------
        基底行列が単位行列になるように解く
        """
        basic_vars_cols = list()
        nonbasic_vars_cols = list()

        # 1次独立な列ベクトルを m 本選ぶ
        for i in range(self.n-1, -1, -1):  # ここどうにかしたい 最後は正順にする
            if len(basic_vars_cols) >= self.m:
                break
            try:
                rank_basis_matrix = np.linalg.matrix_rank(self.basis_matrix)
            except:
                rank_basis_matrix = 0
            insert_rank_basis_matrix = np.linalg.matrix_rank(
                np.insert(self.basis_matrix, 0, self.A[:, i], axis=1)
            )
            if insert_rank_basis_matrix > rank_basis_matrix:
                self.basis_matrix = np.insert(
                    self.basis_matrix, 0, self.A[:, i], axis=1)
                basic_vars_cols.append(i)
            else:
                self.non_basis_matrix = np.insert(
                    self.non_basis_matrix, 0, self.A[:, i], axis=1)
                nonbasic_vars_cols.append(i)

        # 基底行列が単位行列になるようにする
        basis_matrix_inv = np.linalg.inv(self.basis_matrix)
        self.A = np.concatenate([
            np.eye(self.m, self.m), # どうせ逆行列かけたら単位行列になるので
            basis_matrix_inv @ self.non_basis_matrix], # こっちは普通に逆行列かける
            axis=1
        )
        self.b = basis_matrix_inv @ self.b # 普通に逆行列かける
        # 基底変数として選んだインデックスを i1, i2, i3, ...
        # 非基底変数として選んだインデックスを j1, j2, j3, ...
        # とすると、
        # col_swap_list = [..., i3, i2, i1 , ..., j3, j2, j1]
        # A = {..., Ai3, Ai2, Ai1 , ..., Aj3, Aj2, Aj1}
        # となるから、この順にインデックスを振り直すことにする
        # 目的関数の係数ベクトル c のみインデックスの入れ替えがなされていないので、
        # そこを入れ替える
        col_swap_list = basic_vars_cols[::-1] + nonbasic_vars_cols[::-1]
        self.c = self.c[col_swap_list]

    def is_optimized(self) -> bool:
        """
        Notes
        --------
        目的関数が最適化されているか (すなわち、目的関数の係数がすべて非負か)
        を返す
        """
        return np.all(self.c >= 0)

    def next_basic_var(self) -> int:
        """
        Notes
        --------
        目的関数の係数行列の成分のうち、最小のものが次の基底変数のインデックス
        であるから、これを返す
        """
        return np.argmin(self.c)

    def next_nonbasic_var(self) -> int:
        """
        Notes
        --------
        定数部 b を次の基底変数の係数で割り、
        """

if __name__ == "__main__":
    A = np.array([[5, 2, 1, 0, 0], [1, 2, 0, 1, 0], [5, -4, 0, 0, 1]])
    b = np.array([30, 14, 15])
    c = np.array([-5, -4, 0, 0, 0])
    simplexsolver = SimplexSolver(A, b, c)
    simplexsolver.determine_basic_vars()
    # print(simplexsolver.is_optimized())
    # print(simplexsolver.basis_matrix)
    print(simplexsolver.A)
    print(simplexsolver.next_basic_var())
