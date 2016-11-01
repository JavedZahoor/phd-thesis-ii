tree={}
def parent(node):
    for i in tree.keys():
        if node in tree[i]:
            return i
    return False

def PotentialRoot():
    length=0
    root=False
    for i in tree.keys():
        if(len(i.split(';'))>length):
            length=len(i.split(';'))
            root=i
    return root

def child(node):
    try:
        return tree[node]
    except:
        return False

def add(node1,node2):
    if(node1==node2):
        return False
    tree[node1+";"+node2]=[node1,node2]
    return node1+";"+node2

print add("1","0")
print parent("0")
print PotentialRoot()

