# 関数を纏めるクラス
import numpy as np

from steps.step01 import Variable


# 様々なクラスの基幹となるクラス
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()  #必ずオーバーライドせよ！の合図(例外処理)

# Functionを継承
# 継承しているためVariabelを返す必要はない
# 最終的に__call__によって返される
class Square(Function):
    def forward(self, x):
        return x ** 2

if __name__ == "__main__":
    x = Variable(10)
    f = Square()
    y = f(x)
    print(type(y))
    print(y.data)
    print(y)
