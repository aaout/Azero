import numpy as np

from steps.step08 import Exp, Square, Variable


# pythonの関数っぽく使えるように改変
# インスタンス生成と呼び出しを同時に行う
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

# 0次元配列は計算結果がndarrayでなくなるケースがある(p54)
# その対策としてndarrayでないならarrayに変換する関数を設計
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

if __name__ == "__main__":
    x = Variable(np.array(0.5))
    y = square(x)
    print(y.data)

    # 合成関数も使いやすく
    y = square(exp(square(x)))
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
