import unittest

import numpy as np

from steps.step04 import numelical_diff
from steps.step08 import Square, Variable, exp, square


class SquareTest(unittest.TestCase):
    # Square_forward関数のテストケース
    def test_forward(self):
        x = Variable(np.array(2.0)) # 入力
        y = square(x) # 計算結果
        expected = np.array(4.0) # 出力予想値
        self.assertEqual(y.data, expected) # 計算結果と出力予想が一致しているか否か


    # Square_backward関数のテストケース
    def test_backward(self):
        x = Variable(np.array(3.0)) # 入力
        y = square(x)
        y.backward()  # 計算結果
        expected = np.array(6.0) # 出力予想値
        self.assertEqual(x.grad, expected) # 計算結果と出力予想が一致しているか否か

    # 勾配確認
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numelical_diff(square, x)
        # 2値が近いか否か判定
        # allclose(a, b, rtol, atol)
        # 誤差が範囲内に収まっているかどうか
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
