import numpy as np

from steps.step06 import Exp, Square


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    # 変数と関数を紐づける
    # 変数から変数の生みの親である関数を辿ることが出来る
    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad) # xの微分は手前の関数の導関数とdyより求まる
            x.backward() # 再帰

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)  # selfには呼び出した関数が入っている
        self.input = input  # 逆伝播する際に使用するため値を保持
        self.output = output  # この関数から生成された変数を保持する
        return output

    def forward(self, x):
        raise NotImplementedError()  #必ずオーバーライドせよ！の合図(例外処理)

    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        return 2 * x * gy


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        return np.exp(x) * gy


if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    # assert文は条件分岐
    # Falseの場合例外発生→AssertionError
    # 変数と関数を紐づけることで以下の様にたどることが出来る
    assert y.creator == C
    assert y.creator.input == b
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x

    # 自動微分
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
