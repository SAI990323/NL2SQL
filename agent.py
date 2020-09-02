from treelib import Tree
from graph import GraphNode
from util import Queue
from database import TblNode, ColNode, ValNode
from graph import SqlGraphConverter


def interact(sql):
    db = sql.db
    t = Tree()
    t.create_node("root", "root", data=GraphNode("root"))

    q = Queue()
    q.push("root")

    while not q.is_empty():
        t.show()
        cur_id = q.pop()
        cur_node = t[cur_id].data.node
        _pth = [t[i].data.node for i in t.rsearch(cur_id)]
        _pth.reverse()
        pth = []
        for node in _pth:
            n = node if node == "root" else db.get_global_node_id(node)
            pth.append(n)

        child = get_child(pth, cur_node, sql)
        for node in child:
            p_id, c_id = cur_id, str(node)
            c = 1
            while t.contains(c_id):
                c_id = str(node)+str(c)
                c += 1
            t.create_node(identifier=c_id, data=GraphNode(node), parent=p_id)
            q.push(c_id)
    return t


def get_child(pos, node, sql, gold=True):
    if gold: #???
        return get_child_gold(pos, sql)
    else:
        return get_child_pred(pos, node, sql)


def get_child_gold(pos, sql):
    db = sql.db
    t = SqlGraphConverter.sql2graph(sql)
    nid = locate_pos(pos, t, db)
    c = [e.data.node for e in t.children(nid)]
    return c


def get_child_pred(pos, node, sql):

    if node == "root":
        return "pred node"
    elif type(node) == TblNode:
        return "pred node"
    elif type(node) == ColNode:
        return "pred node"
    else:
        assert False, "Bad node Type: " + str(type(node))


def ask_user(pos, node, sql): #???
    return True


def locate_pos(pos, t, db):
    cur_id = "root"
    for i in range(1, len(pos)):
        c = t.children(cur_id)
        found = False
        for e in c:
            c_id = db.get_global_node_id(e.data.node) if e.data.node != "root" else "root"
            if c_id == pos[i]:
                cur_id = e.identifier
                found = True
                break
        assert found, "Bad Case."
    return cur_id

