import numpy as np
datap=np.load("p.npy")
datay=np.load("y.npy")

def p_po(data):
    k=0
    for i,row in enumerate(data):
        for j,col in enumerate(row):
            if data[i,j,1] >= 9.97236181e-01:
                k=k+1
    return k

def y_po(data):
    k=0
    for i,row in enumerate(data):
        for j,col in enumerate(row):
            # if data[i,j] >= 0:
            if data[i,j] == 1:
                k=k+1
    return k

def p_ne(data):
    k=0
    for i,row in enumerate(data):
        for j,col in enumerate(row):
            if data[i,j,1] < 9.97236181e-01:
                k=k+1
    return k

def y_ne(data):
    k=0
    for i,row in enumerate(data):
        for j,col in enumerate(row):
            # if data[i,j] >= 0:
            if data[i,j] == 0:
                k=k+1
    return k

# 一文の中でWWを検出した割合  
def p_cen(data):
    k=0
    for i,row in enumerate(data):
        for j,col in enumerate(row):
            if data[i,j,1] >= 9.97236181e-01:
                k=k+1 
                break 
    return k

print("y.shape(正解):")
print(datay.shape)
print("p.shape(予測):")
print(datap.shape)

nump_po=p_po(datap)
numy_po=y_po(datay)
print("予測ラベル(1)の数:")
print(nump_po)
print("正解ラベル(1)の数:")
print(numy_po)

nump_ne=p_ne(datap)
numy_ne=y_ne(datay)
print("予測ラベル(0)の数:")
print(nump_ne)
print("正解ラベル(0)の数:")
print(numy_ne)

print("全ラベル数:")
sum_label=datay.shape[0]*datay.shape[1]
print(sum_label)

print("正解ラベルが1のうち予測ラベルが1の割合（再現率:Recall）:")
print(str((nump_po/numy_po)*100)+"%")
nump_cen=p_cen(datap)
print("WWを検出した文の数:")
print(nump_cen)
i=0
for row in datap:
    i=i+1

print("全文の数:")
print(i)
print("全文のうちwwが含まれている文を検出した割合:")
print(str((nump_cen/i)*100)+"%")

print("0を1と間違った数:")
false_ne=numy_ne*6.39034760e-04
# false_ne=sum_label*6.39034760e-04
print(false_ne)

print("1を0と間違った数:")
false_po=numy_po*1.16283186e-01
print(false_po)


print("ネガティブ区間の総時間(s):")
frame_s=0.032
ne_time=numy_ne*frame_s
print(ne_time)

print("FAh(1時間あたりに誤って Wake Word を誤検出した回数:)")
FAh=(false_ne/ne_time)*3600
print(FAh)
print(FAh*frame_s)

