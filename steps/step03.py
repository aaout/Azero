import numpy as np

from steps.step02 import Function, Square, Variable


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

if __name__ == "__main__":
    # 関数や変数の型をそろえることで連結が容易になる
    A = Square()
    B = Exp()
    C = Square()
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    c = C(b)
    print(type(a))
    print(type(b))
    print(type(c))
