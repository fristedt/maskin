import monkdata as m
import dtree as d
import random
# import drawtree

def assignment1():
    print("Assignment 1")
    print("MONK-1: %f\nMONK-2: %f\nMONK-3: %f" % tuple([d.entropy(x) for x in (m.monk1, m.monk2, m.monk3)]))
    print("Note: The impurity of MONK-1 is 1. This can only happen when there is an equal amount of true and false samples in the data.")

def assignment2():
    print("Assignment 2")
    l = [[d.averageGain(m.monk1, a) for a in m.attributes],
            [d.averageGain(m.monk2, a) for a in m.attributes],
            [d.averageGain(m.monk3, a) for a in m.attributes]]
    for i in range(0, 3):
        # print("MONK-%d: %f %f %f %f %f %f" % tuple([i + 1] + l[i]), end=" | ")
        print("Split on: a%d" % (max(enumerate(l[i]), key=lambda p: p[1])[0] + 1))

def assignment3(): 
    for dataset in [(m.monk1, m.monk1test), (m.monk2, m.monk2test), (m.monk3, m.monk3test)]:
        tree = d.buildTree(dataset[0], m.attributes)
        # [print(d.check(tree, dataset[i])) for i in range(0, 2)]

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def assignment4(): 
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    dataset = [("monk1", m.monk1, m.monk1test), ("monk3", m.monk3, m.monk3test)]
    l = []
    for f in fractions:
        extraCurry = d.check(assignment4helper(dataset[0][1], f), dataset[0][2])
        stektLok = d.check(assignment4helper(dataset[1][1], f), dataset[1][2])
        print("%.2f %.2f %.2f" % (f, 1 - extraCurry, 1 - stektLok))
        l.append(extraCurry)
    # maximum = max(l)
    # maxIndices = [i for i, j in enumerate(l) if j == maximum]
    # for i in maxIndices:
    #     print("Best fraction %.1f" % fractions[i])
        
def assignment4helper(dataset, fraction):
    monk1train, monk1val = partition(dataset, fraction)
    tree = d.buildTree(monk1train, m.attributes)

    bestTree = None
    maxVal = -1
    cont = True
    i = 0
    while (cont):
        cont = False
        i += 1
        for t in d.allPruned(tree):
            val = d.check(t, monk1val)
            if (val > maxVal):
                cont = True
                bestTree = t
                maxVal = val
        tree = bestTree
    # print("#iterations: %d" % i)
    return tree

    # print("bestTree:    %s" % bestTree)
    # print("value:       %.2f" % maxVal)


# assignment1()
# assignment2()
# assignment3()
assignment4()

def caspersky(dataset):
    print("Assignment 3")
    a = d.bestAttribute(dataset, m.attributes)
    branches = []
    for v in a.values:
        s = d.select(dataset, a, v)
        tf = d.mostCommon(s)
        if tf == True:
            branches.append((v, d.TreeLeaf(s)))
        else:
            a2 = d.bestAttribute(s, m.attributes)
            branches2 = []
            for v2 in a2.values:
                s2 = d.select(s, a2, v2)
                branches2.append((v2, d.TreeLeaf(d.mostCommon(s2))))
            branches.append((v, d.TreeNode(a2, dict(branches2), d.mostCommon(s))))
    
    drawtree.drawTree(d.TreeNode(a, dict(branches), d.mostCommon(dataset)))
    # drawtree.drawTree(d.buildTree(m.monk1, m.attributes, 1))

# caspersky(m.monk1)
