import time


def timef():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def tprint(msg):
    print(timef(), end=": ")
    print(msg)

def group2(x):
    for i in range(0,len(x),2):
        if i+1 >= len(x):
            yield (x[i],)
        else:
            yield (x[i], x[i+1])
