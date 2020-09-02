from treelib import Tree
from sql import SelectClause, FromClause, FilterClause, GroupClause, OrderClause


class NodeAttr(object):
    def __init__(self):
        self.kw = "none"

    def get_attr(self):
        attr = ["op", "agg", "dist", "limit", "trend", "conj"]
        return [self.__dict__[i] for i in self.__dict__ if i in attr]

    def __str__(self):
        attr_items = ["kw", "op", "agg", "dist", "limit", "trend", "conj"]
        res = []
        for i in attr_items:
            if i not in self.__dict__:
                continue
            res.append("%s(%s)" % (i, self.__dict__[i]))
        return " ".join(res)


class SelectAttr(NodeAttr):
    def __init__(self):
        super(SelectAttr, self).__init__()
        self.agg = "none"
        self.dist = False


class FilterAttr(NodeAttr):
    def __init__(self):
        super(FilterAttr, self).__init__()
        self.agg = "none"


class GroupAttr(NodeAttr):
    def __init__(self):
        super(GroupAttr, self).__init__()
        self.agg = "none"


class OrderAttr(NodeAttr):
    def __init__(self):
        super(OrderAttr, self).__init__()
        self.agg = "none"
        self.limit = False
        self.trend = "asc"


class ValueAttr(NodeAttr):
    def __init__(self):
        super(ValueAttr, self).__init__()
        self.conj = "and"
        self.op = "none"


class GraphNode(object):
    def __init__(self, db_node):
        self.node = db_node
        self.__kw = []
        self.__attr = []
        self.id = str(db_node) + " "

    def add_attr(self, attr: NodeAttr):
        self.__kw.append(attr.kw)
        self.__attr.append(attr)
        self.id += str(attr)

    def get_kws(self):
        return self.__kw

    def get_attrs(self):
        return [i.get_attr() for i in self.__attr]


class SqlGraphConverter(object):
    @classmethod
    def sql2graph(cls, sql):
        g = Tree()
        g.create_node("root", "root", data=GraphNode("root"))

        from_attrs = cls.__get_attr(sql.frm)
        for tbl, attr in zip(sql.frm.get_sql_tbls(), from_attrs):
            tbl_id = str(tbl)
            g.create_node(identifier=tbl_id, data=GraphNode(tbl), parent="root")
            g.get_node(tbl_id).data.add_attr(attr)

        for e in [sql.select, sql.group, sql.order]:
            if e is None:
                continue
            attrs = cls.__get_attr(e)
            for col, attr in zip(e.get_nodes(), attrs):
                col_id = str(col)
                tbl_id = str(col.get_tbl()) if col_id != "*(col-0)" else "root" #???
                if not g.get_node(col_id):
                    g.create_node(identifier=col_id, data=GraphNode(col), parent=tbl_id)

                g.get_node(col_id).data.add_attr(attr)

        for e in [sql.where, sql.having]:
            if e is None:
                continue
            attrs = cls.__get_attr(e)
            for col, val, col_attr, val_attr in zip(e.get_nodes(), e.get_values_node(), *attrs):
                col_id, val_id = str(col), str(val)
                tbl_id = str(col.get_tbl()) if col_id != "*(col-0)" else "root"

                if not g.get_node(col_id):
                    g.create_node(identifier=col_id, data=GraphNode(col), parent=tbl_id)
                g.get_node(col_id).data.add_attr(col_attr)

                if not g.get_node(val_id):
                    g.create_node(identifier=val_id, data=GraphNode(val), parent=col_id)
                g.get_node(val_id).data.add_attr(val_attr)

        return g

    @classmethod
    def graph2sql(cls, g): #???
        return 0

    @classmethod
    def __get_attr(cls, clause):
        clause_type = type(clause)

        # switch clause
        if clause_type == SelectClause:
            return cls.__convert_sel(clause)
        elif clause_type == FromClause:
            return cls.__convert_from(clause)
        elif clause_type == FilterClause:
            return cls.__convert_filter(clause)
        elif clause_type == GroupClause:
            return cls.__convert_group(clause)
        elif clause_type == OrderClause:
            return cls.__convert_order(clause)
        else:
            assert False, "Unknown Clause Type Error"

    @classmethod
    def __convert_sel(cls, clause: SelectClause):
        clause_attrs = clause.get_attrs()
        node_attrs = [SelectAttr() for _ in range(clause.column_num)]
        for i in range(clause.column_num):
            node_attrs[i].kw = clause.kw
            node_attrs[i].agg = clause_attrs["agg"][i]
            node_attrs[i].dist = clause_attrs["dist"][i]
        return node_attrs

    @classmethod
    def __convert_from(cls, clause: FromClause):
        node_attrs = [NodeAttr() for _ in range(clause.get_tbl_num())]
        for i in node_attrs:
            i.kw = "from"
        return node_attrs

    @classmethod
    def __convert_filter(cls, clause: FilterClause):
        clause_attrs = clause.get_attrs()
        node_attrs = [FilterAttr() for _ in range(clause.condition_num)]
        value_attrs = [ValueAttr() for _ in range(clause.condition_num)]
        for i in range(clause.condition_num):
            node_attrs[i].kw = clause.kw
            node_attrs[i].agg = clause_attrs["agg"][i]
            value_attrs[i].kw = "op"
            value_attrs[i].op = clause_attrs["op"][i]
            value_attrs[i].conj = clause_attrs["conj"][i]
        return node_attrs, value_attrs

    @classmethod
    def __convert_group(cls, clause: GroupClause):
        clause_attrs = clause.get_attrs()
        node_attrs = [GroupAttr() for _ in range(clause.group_num)]
        for i in range(clause.group_num):
            node_attrs[i].kw = clause.kw
            node_attrs[i].agg = clause_attrs["agg"][i]
        return node_attrs

    @classmethod
    def __convert_order(cls, clause: OrderClause):
        clause_attrs = clause.get_attrs()
        node_attrs = [OrderAttr() for _ in range(clause.order_num)]
        for i in range(clause.order_num):
            node_attrs[i].kw = clause.kw
            node_attrs[i].agg = clause_attrs["agg"][i]
            node_attrs[i].limit = clause_attrs["limit"]
            node_attrs[i].trend = clause_attrs["trend"]
        return node_attrs
