from numpy import *
from numpy import linalg as la


def loadExData():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]


data = mat(loadExData())


def ecludSim(inA,inB):
    return 1.0/(1.0+la.norm(inA-inB)) #inA，inB是列向量


def pearSim(inA,inB):
#    print(inA,len(inA))
    if len(inA) < 3:
        return 1.0
    else:
#        print("corrcoef(inA,inB,rowvar=0)",corrcoef(inA,inB,rowvar=0))
        #对称矩阵，且corrcoef是x1x1，x1x2,x2x1,x2x2这四者系数。
#        print("corrcoef(inA,inB,rowvar=0)[0]",corrcoef(inA,inB,rowvar=0)[0])
        #由于两个变量，所以取第一行就是x1对所有变量的线性相关性，协方差。
#        print("corrcoef(inA,inB,rowvar=0)[0][1]",corrcoef(inA,inB,rowvar=0)[0][1])
        #第一行第二列就是x2x1，第二列和第二行一样都是第二个变量对所有其他变量的线性相关性。
        return 0.5+0.5*corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA,inB):#inA,inB是列向量
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)


print("ecludSim=", ecludSim(data[:, 1], data[:, 2]))
print("pearSim=", pearSim(data[:, 1], data[:, 2]))
print("cosSim=", cosSim(data[:, 1], data[:, 2]))