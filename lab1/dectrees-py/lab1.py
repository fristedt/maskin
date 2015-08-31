import monkdata as m
import dtree as d

def assignment1():
    print("Assignment 1")
    print("MONK-1: %f\nMONK-2: %f\nMONK-3: %f" % tuple([d.entropy(x) for x in (m.monk1, m.monk2, m.monk3)]))

def assignment2():
    print("Assignment 2")
    l = [[d.averageGain(m.monk1, a) for a in m.attributes],
            [d.averageGain(m.monk2, a) for a in m.attributes],
            [d.averageGain(m.monk3, a) for a in m.attributes]]
    for i in range(0, 3):
        print("MONK-%d: %f %f %f %f %f %f" % tuple([i + 1] + l[i]), end=" | ")
        print("Max: a%d" % (max(enumerate(l[i]), key=lambda p: p[1])[0] + 1))

def assignment3(): 
    print("Assignment 3")
    t=d.buildTree(m.monk1, m.attributes, 2)
    print(t)
    print(d.check(t, m.monk1test))

assignment1()
assignment2()
assignment3()

