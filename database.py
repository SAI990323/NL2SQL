
class NodeType(object):
    COL = 'col'
    TBL = 'tab'
    VAL = 'val'
    VIR = "Virtual"


class DatabaseNode(object):
    FULL_NAME_SEPARATOR = " # "

    def __init__(self, node_id: int,
                 node_type: str,
                 node_name: str,
                 origin_name: str,
                 db):

        self.__node_id: int = node_id
        self.__node_type: str = node_type
        self.__node_name: str = node_name
        self.__origin_name: str = origin_name  # default
        self.__db = db

    def get_node_id(self):
        return self.__node_id

    def get_node_type(self):
        return self.__node_type

    def get_node_name(self):
        return self.__node_name

    def get_node_origin_name(self):
        return self.__origin_name

    def get_db(self):
        return self.__db

    def get_node_fingerprint(self):
        fingerprint = "%s(%s-%d)" % (self.__node_name, self.__node_type, self.__node_id)
        return fingerprint

    def __str__(self):
        return self.get_node_fingerprint()


class TblNode(DatabaseNode):
    def __init__(self, node_id: int,
                 node_type: str,
                 node_name: str,
                 origin_name: str,
                 db):
        DatabaseNode.__init__(self, node_id, node_type, node_name, origin_name, db)
        self.__primary_key = None

    def get_full_name(self):
        return self.get_node_name()

    def set_primary_key(self, col_node):
        self.__primary_key = col_node

    def get_primary_key(self):
        return self.__primary_key


class ColNode(DatabaseNode):
    def __init__(self, node_id: int,
                 node_type: str,
                 node_name: str,
                 origin_name: str,
                 tbl_node: TblNode,
                 db):
        DatabaseNode.__init__(self, node_id, node_type, node_name, origin_name, db)
        self.tbl_node = tbl_node

    def get_tbl(self):
        return self.tbl_node

    def get_full_name(self):
        name = self.get_node_name()
        if self.get_node_id():
            name = self.tbl_node.get_full_name() + self.FULL_NAME_SEPARATOR + name
        return name


class ValNode(DatabaseNode):
    def __init__(self, node_id: int,
                 node_type: str,
                 node_name: str,
                 origin_name: str,
                 col_node: ColNode,
                 db):
        DatabaseNode.__init__(self, node_id, node_type, node_name, origin_name, db)
        self.col_node = col_node

    def get_col(self):
        return self.col_node

    def get_tbl(self):
        return self.col_node.get_tbl()

    def get_full_name(self):
        return self.col_node.get_full_name() + self.FULL_NAME_SEPARATOR + \
            self.get_node_name()


class Database(object):
    # Zero denotes None
    Attribute = 1 #???
    Primary = 2
    Foreign = 3
    VIRTUAL_NODE = "virtual" #???
    VIRTUAL_NODE_TYPE = "virtual"

    def __init__(self, db_dict):
        db_id = db_dict["db_id"].lower()
        foreign_keys = db_dict["foreign_keys"]
        primary_keys = db_dict["primary_keys"]
        column_names = db_dict["column_names"]
        column_names_original = db_dict["column_names_original"]
        table_names = db_dict["table_names"]
        table_names_original = db_dict["table_names_original"]

        self.db_id = db_id

        self.tbl_nodes = []
        for i in range(len(table_names)):
            tbl_name = table_names[i]
            tbl_name_ori = table_names_original[i]
            db_node = TblNode(i, NodeType.TBL, tbl_name.lower(), tbl_name_ori.lower(), self)
            self.tbl_nodes.append(db_node)

        self.col_tbl = []
        self.col_nodes = []
        for i in range(len(column_names)):
            tbl_id, col_name = column_names[i]
            tbl_id, col_name_ori = column_names_original[i]
            tbl_node = self.tbl_nodes[tbl_id]
            db_node = ColNode(i, NodeType.COL, col_name.lower(), col_name_ori.lower(), tbl_node, self)
            self.col_nodes.append(db_node)
            self.col_tbl.append(tbl_id)

        vir_node_id = len(self.tbl_nodes)
        virtual_node = TblNode(vir_node_id, NodeType.VIR, Database.VIRTUAL_NODE, Database.VIRTUAL_NODE, self)
        self.tbl_nodes.append(virtual_node)

        self.tbl_names = table_names
        self.foreign = foreign_keys
        self.primary = primary_keys

        self.col_num = len(self.col_nodes)
        self.tbl_num = len(self.tbl_nodes)
        self.node_num = self.col_num + self.tbl_num

        self.fa = [i for i in range(self.col_num + self.tbl_num)]

    def get_global_id(self, _id, _type):
        # if _type == NodeType.TBL:
        #     _id += self.col_num
        if _type == NodeType.COL:
            _id += self.tbl_num
        elif _type == NodeType.VAL:
            _id += self.tbl_num
        return _id

    def get_node(self, node_id: int, node_type: str, tok=""):
        if node_type == NodeType.COL:
            node = self.col_nodes[node_id]

        elif node_type == NodeType.TBL:
            node = self.tbl_nodes[node_id]

        elif node_type == NodeType.VIR:
            node = self.tbl_nodes[node_id]

        elif node_type == NodeType.VAL:
            assert tok, "Value Node needs value"
            node = self.__get_value_node(node_id, node_type, tok)

        else:
            assert False, "Unknown Node Type: %s" % node_type

        return node

    def get_node_global(self, g_id):
        return (self.tbl_nodes + self.col_nodes)[g_id]

    def __get_value_node(self, node_id: int, node_type: str, value):
        col_node = self.get_node(node_id, NodeType.COL)
        node_id = col_node.get_node_id()
        val_node = ValNode(node_id, node_type, value, value, col_node, self)
        return val_node

    def get_global_node_id(self, node: DatabaseNode):
        if node.get_node_type() == NodeType.COL:
            global_id = node.get_node_id() + self.tbl_num
        elif node.get_node_type() == NodeType.VAL:
            global_id = node.get_node_id() + self.tbl_num
        else:
            global_id = node.get_node_id()

        return global_id

    def parse_global_node_id(self, g_id):
        node = self.get_node_global(g_id)
        return node.get_node_id(), node.get_node_type()

    def __str__(self):
        return "DB: {}".format(self.db_id)
