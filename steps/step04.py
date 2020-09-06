import numpy as np

from steps.step03 import Exp, Function, Square, Variable


# f: 関数  x: 微分するx座標
def numelical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / 2 * eps

if __name__ == "__main__":
    x = Variable(np.array(2))
    f = Square()
    dy = numelical_diff(f, x)
    print(dy)

    # 合成関数
    def f(x):
        A = Square()
        B = Exp()
        C = Square()
        return C(B(A(x)))

    # 合成関数の微分
    x = Variable(np.array(0.5))
    dy = numelical_diff(f, x)
    print (dy)
