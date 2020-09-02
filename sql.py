from database import NodeType
from database import DatabaseNode
from database import TblNode
from database import ColNode


class TableAlias(object):
    def __init__(self):
        self.__tbls = []
        self.__alias = {}

    def add_tbl(self, tbl_node: TblNode):
        if self.__exist_alias(tbl_node):
            return 0

        self.__tbls.append(tbl_node)

        alias = "T%d" % len(self.__tbls)
        key = tbl_node.get_node_origin_name()

        self.__alias[key] = alias

    def get_tbls(self):
        return self.__tbls

    def get_alias(self, tbl_node: TblNode):
        assert type(tbl_node) == TblNode, "Expect TblNode instance, but got %s" % str(str(tbl_node))
        self.add_tbl(tbl_node)

        key = tbl_node.get_node_origin_name()
        return self.__alias[key]

    def __exist_alias(self, tbl_node: TblNode):
        key = tbl_node.get_node_origin_name()
        return key in self.__alias


class SqlClause(object):
    def __init__(self):
        self.__db_node = []
        self.__tbl_alias = TableAlias()

    def add_node(self, node: DatabaseNode):
        self.__db_node.append(node)

    def get_nodes(self):
        return self.__db_node

    def get_attrs(self):
        return {}

    def set_alias(self, tbl_alias: TableAlias):
        self.__tbl_alias = tbl_alias

    def get_node_alias(self, node):
        if type(node) == TblNode:
            alias = self.__tbl_alias.get_alias(node)

        elif type(node) == ColNode:

            # iff column is '*' and need alias (sql has more than one tables)
            if node.get_node_id() and self.need_alias():
                name = node.get_node_origin_name()
                tbl_alias = self.__tbl_alias.get_alias(node.get_tbl())
                alias = "%s.%s" % (tbl_alias, name)
            else:
                alias = node.get_node_origin_name()

        else:
            assert False, "Error: Unknown node type %s" % (str(type(node)))

        return alias

    def get_sql_tbls(self):
        return self.__tbl_alias.get_tbls()

    def need_alias(self):
        return len(self.__tbl_alias.get_tbls()) > 1


class SelectClause(SqlClause):
    kw = "select"

    def __init__(self):
        SqlClause.__init__(self)
        self.column_num = 0
        self.__agg = []
        self.__distinction = []

    def add_column(self, node: DatabaseNode, agg: str, is_dist: bool):
        self.add_node(node)
        self.__agg.append(agg)
        self.__distinction.append(is_dist)
        self.column_num += 1

    def get_attrs(self):
        return {
            "agg": self.__agg,
            "dist": self.__distinction
        }

    def __str__(self):
        res = []
        for i in range(self.column_num):
            col_str = self.get_node_alias(self.get_nodes()[i])
            aggregation = self.__agg[i]
            is_dist = self.__distinction[i]

            if is_dist:
                col_str = "distinct %s" % col_str

            if aggregation != 'none':
                col_str = "%s(%s)" % (aggregation, col_str)

            res.append(col_str)
        res = "SELECT %s" % (", ".join(res))

        return res


class FromClause(SqlClause):
    kw = "from"

    def __init__(self):
        SqlClause.__init__(self)
        self.table_num = 0

    def add_table(self, node):
        self.add_node(node)
        self.table_num += 1
        pass

    def get_tbl_num(self):
        return len(self.get_sql_tbls())

    def __str__(self):
        tbls = self.get_sql_tbls()
        table_num = len(tbls)

        res = []
        for i in range(table_num):
            tbl_node = tbls[i]
            tbl_str = tbl_node.get_node_origin_name()

            if self.need_alias():
                alias = self.get_node_alias(tbl_node)
                tbl_str = "%s AS %s" % (tbl_str, alias)

            res.append(tbl_str)

        res = "FROM %s" % " join ".join(res)
        return res


class FilterClause(SqlClause):
    kw = "filter"

    def __init__(self, clause):
        SqlClause.__init__(self)
        self.condition_num = 0
        self.__agg = []
        self.__op = []
        self.__value = []
        self.__conj = []
        self.clause = clause

    def add_condition(self, node, agg, op, value, conj):
        self.add_node(node)
        self.__agg.append(agg)
        self.__op.append(op)
        self.__value.append(value)
        self.__conj.append(conj)
        self.condition_num += 1

    def get_values(self):
        return self.__value

    def get_values_node(self):
        val_nodes = []
        for node, val in zip(self.get_nodes(), self.__value):
            db = node.get_db()
            val_node = db.get_node(node.get_node_id(), NodeType.VAL, str(val))
            val_nodes.append(val_node)
        return val_nodes

    def get_attrs(self):
        return {
            "op": self.__op,
            "agg": self.__agg,
            "conj": self.__conj,
            "value": self.__value
        }

    def __str__(self):

        if not self.condition_num:
            return ""

        res = self.clause
        for i in range(self.condition_num):

            col_str = self.get_node_alias(self.get_nodes()[i])
            aggregation = self.__agg[i]

            if aggregation != 'none':
                col_str = "%s(%s)" % (aggregation, col_str)

            op = self.__op[i]
            value = self.__value[i]
            conj = self.__conj[i]

            if type(value) == SqlQuery:
                value = "(%s)" % str(value)

            elif type(value) == str:
                value = value

            else:
                value = "\"%s\"" % value

            col_str = "%s %s %s" % (col_str, op, value)
            if op == "between":
                col_str = "%s and \"value\"" % col_str

            res = "%s %s" % (res, col_str) if i == 0 else "%s %s %s" % (res, conj, col_str)

        return res


class GroupClause(SqlClause):
    kw = "group by"

    def __init__(self):
        SqlClause.__init__(self)
        self.__agg = []
        self.group_num = 0

    def add_group(self, node, agg):
        self.add_node(node)
        self.__agg.append(agg)
        self.group_num += 1

    def get_attrs(self):
        return {
            "agg": self.__agg
        }

    def __str__(self):
        if not self.group_num:
            return ""

        res = []
        for i in range(self.group_num):
            col_str = self.get_node_alias(self.get_nodes()[i])
            aggregation = self.__agg[i]

            if aggregation != 'none':
                col_str = "%s(%s)" % (aggregation, col_str)
            res.append(col_str)

        res = "GROUP BY %s" % ", ".join(res)
        return res


class OrderClause(SqlClause):
    kw = "order by"

    def __init__(self):
        SqlClause.__init__(self)
        self.__agg = []
        self.__limit = 0
        self.__trend = ''
        self.order_num = 0

    def add_order(self, node: DatabaseNode, agg: str):
        self.add_node(node)
        self.__agg.append(agg)
        self.order_num += 1

    def set_limit(self, limit):
        self.__limit = limit

    def set_trend(self, trend):
        self.__trend = trend

    def get_attrs(self):
        return {
            "agg": self.__agg,
            "limit": self.__limit,
            "trend": self.__trend
        }

    def __str__(self):
        if not self.order_num:
            return ""

        res = []
        for i in range(self.order_num):
            col_str = self.get_node_alias(self.get_nodes()[i])
            aggregation = self.__agg[i]
            if aggregation != 'none':
                col_str = "%s(%s)" % (aggregation, col_str)
            res.append(col_str)

        res = "ORDER BY %s" % ", ".join(res)
        if self.__trend:
            res = "%s %s" % (res, self.__trend)

        if self.__limit:
            res = "%s LIMIT %d" % (res, self.__limit)

        return res


class SqlQuery(object):
    def __init__(self, select, frm, where, group, having, order, intersect, union, exc, db): #???
        self.select = select
        self.frm = frm
        self.where = where
        self.group = group
        self.having = having
        self.order = order
        self.intersect = intersect
        self.union = union
        self.exc = exc
        self.db = db

    def __str__(self):
        comps = [
            self.select,
            self.frm,
            self.where,
            self.group,
            self.having,
            self.order,
            self.intersect,
            self.union,
            self.exc
        ]

        main = [str(comp) for comp in comps if str(comp) and comp is not None]
        main = " ".join(main)

        return main


class IUESubQuery(SqlQuery): #???
    def __init__(self, kw, select, frm, where, group, having, order, intersect, union, exc, db):
        SqlQuery.__init__(self, select, frm, where, group, having, order, intersect, union, exc, db)
        self.kw = kw

    def __str__(self):
        comps = [
            self.select,
            self.frm,
            self.where,
            self.group,
            self.having,
            self.order,
            self.intersect,
            self.union,
            self.exc
        ]

        main = [str(comp) for comp in comps if str(comp) and comp is not None]
        main = "%s %s" % (self.kw, " ".join(main))
        return main


# For loading original data (merge to training data)
class KeywordStr(object):
    CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
    SQL_OPS = ('intersect', 'union', 'except')
    JOIN_KEYWORDS = ('join', 'on', 'as')
    COND_OPS = ('and', 'or')
    ORDER_OPS = ('desc', 'asc')
    WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
    UNIT_OPS = ('none', '-', '+', "*", '/')
    AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
    TABLE_TYPE = {
        'sql': "sql",
        'table_unit': "table_unit",
    }

    @classmethod
    def get_agg(cls, index):
        return cls.AGG_OPS[index]

    @classmethod
    def get_clause(cls, index):
        return cls.CLAUSE_KEYWORDS[index]

    @classmethod
    def get_filter_op(cls, index):
        return cls.WHERE_OPS[index]


# For training data
class SqlKeyWords(object):
    END_TOK = "<eos>" #???
    SEP_TOK = "<sep>" #???
    clause = ['select', 'from', 'filter', 'where', 'op', 'group by', "having", 'order by', 'limit', 'intersect', 'union', 'except']
    col_to_col = ["=", ">", "<", ">=", "<=", "!=", "like", "is", "exists", "between", "in", "not in", "limit"]
    tab_to_tab = ["join"]
    agg = ['none', 'max', 'min', 'count', 'sum', 'avg']
    special_kw = ["root", SEP_TOK, END_TOK]#???

    keywords = agg + tab_to_tab + col_to_col + clause + special_kw
    kw_num = len(keywords)

    @classmethod
    def get_index(cls, kw: str):
        kw = cls.unify(kw)

        if kw not in cls.keywords:
            assert False, "Unknown keyword: %s" % kw
        return cls.keywords.index(kw)

    @classmethod
    def get_str(cls, index: int):
        return cls.keywords[index]

    @classmethod
    def unify(cls, kw):

        if kw == "orderBy":
            unified_kw = 'order by'

        elif kw == "groupBy":
            unified_kw = 'group by'

        else:
            unified_kw = kw

        return unified_kw.lower()


class SqlQueryLoader(object):

    @classmethod
    def load(cls, sql_json, db):
        comps = cls.__parse(sql_json, db)
        comps = cls.__update_tbl_alias(comps)  # table is not completed in origin data ???
        return SqlQuery(*comps)

    @classmethod
    def __load_iue_subquery(cls, sql_json, db, kw): #???
        comps = cls.__parse(sql_json, db)
        comps = cls.__update_tbl_alias(comps)
        return IUESubQuery(kw, *comps)

    @classmethod
    def __parse(cls, sql_json, db):
        sel_clause = cls.__load_select(sql_json["select"], db)
        from_clause = cls.__load_from(sql_json["from"], db)
        where_clause = cls.__load_filter(sql_json["where"], db, "WHERE")
        group_clause = cls.__load_group(sql_json["groupBy"], db)
        having_clause = cls.__load_filter(sql_json["having"], db, "HAVING")
        order_clause = cls.__load_order(sql_json["orderBy"], sql_json["limit"], db)
        except_clause = cls.__load_iue(sql_json["except"], db, "EXCEPT")
        union_clause = cls.__load_iue(sql_json["union"], db, "UNION")
        intersect_clause = cls.__load_iue(sql_json["intersect"], db, "INTERSECT")

        return [sel_clause, from_clause, where_clause, group_clause, having_clause, order_clause,
                intersect_clause, union_clause, except_clause, db]

    @classmethod
    def __load_select(cls, json_data, db):
        # Format @ select: [isdistinct, [Agg_ops, [Unit_ops, [inner_Agg_ops, icol, isdistinct], op2]], ... ] ???
        isdistinct, columns = json_data

        sel = SelectClause()
        for idx, sel_unit in enumerate(columns):
            agg_id, col_unit = sel_unit
            agg = KeywordStr.get_agg(agg_id)

            col_id, inner_agg, is_dist = cls.__parse_col_unit(col_unit)
            assert inner_agg == 'none', "Error"

            node = db.get_node(col_id, NodeType.COL)

            sel.add_column(node, agg, is_dist)
        return sel

    @classmethod
    def __load_filter(cls, json_data, db, kw):
        # Part IV: σ (Where and Having)
        # [is_not, op, [Unit_ops, [inner_Agg_ops, col_id, isdistinct], op2], Val1, Val2]
        #  - is_not (0), inner_Agg_ops (!0 but 0->3), op2 (None)

        assert kw.lower() in ['where', 'having'], "Error:illegal filter keyword %s" % kw
        fltr = FilterClause(kw)

        conj = "and"  # default and
        for cond_unit in json_data:

            if cond_unit in ["and", "or"]:
                conj = cond_unit
                continue

            [is_not, op_id, col_unit, value, _] = cond_unit
            col_id, agg, is_dist = cls.__parse_col_unit(col_unit)

            op = KeywordStr.get_filter_op(op_id)
            if is_not:
                op = "not %s" % op

            if type(value) == dict:
                value = cls.load(value, db)
            else:
                value = value

            node = db.get_node(col_id, NodeType.COL)
            fltr.add_condition(node, agg, op, value, conj)

        return fltr

    @classmethod
    def __load_order(cls, json_data, limit, db):
        # Part V: orderBy
        # ["asc"/"desc", [[Unit_ops, [inner_Agg_ops, icol, isdistinct], op2], ...]]
        order = OrderClause()

        if not json_data:
            return order

        trend, columns = json_data
        order.set_trend(trend)

        for order_unit in columns:
            col_id, agg, is_dist = cls.__parse_col_unit(order_unit)
            node = db.get_node(col_id, NodeType.COL)
            order.add_order(node, agg)

        if limit:
            order.set_limit(limit)

        return order

    @classmethod
    def __load_iue(cls, json_data, db, kw):
        # Part VI: Intersection, Union and Exception
        if json_data is None:
            return None
        return cls.__load_iue_subquery(json_data, db, kw)

    @staticmethod
    def __load_from(json_data, db):
        # Part II From: table -> column -> column -> table -> ...
        f = FromClause()

        for unit_type, unit in json_data["table_units"]:
            if unit_type == "table_unit":
                tbl_id = unit
                node = db.get_node(tbl_id, NodeType.TBL)
                f.add_table(node)
            elif unit_type == "sql":
                _ = 1

        return f

    @staticmethod
    def __load_group(json_data, db):
        # Part III: Group By
        # [[Agg_ops, col_id, isdistinct], ...]
        #  - inner_Agg_ops (0), col_id(!0), isdistinct(0)

        group = GroupClause()
        for group_unit in json_data:
            agg_id, col_id, is_dist = group_unit
            agg = KeywordStr.get_agg(agg_id)
            node = db.get_node(col_id, NodeType.COL)
            group.add_group(node, agg)
        return group

    @staticmethod
    def __update_tbl_alias(comps):
        tbl_alias = TableAlias()

        sel_clause, from_clause, where_clause, group_clause, \
            having_clause, order_clause, intersect_clause, union_clause, except_clause, db = comps

        col_based_clause = [sel_clause, where_clause, group_clause, having_clause, order_clause]
        tbl_based_clause = [from_clause]

        for clause in col_based_clause:
            for node in clause.get_nodes():
                tbl_node = node.get_tbl()
                if node.get_node_id():  # iff the column is not "*"
                    tbl_alias.add_tbl(tbl_node)

        for clause in tbl_based_clause:
            for tbl_node in clause.get_nodes():
                tbl_alias.add_tbl(tbl_node)

        sel_clause.set_alias(tbl_alias)
        from_clause.set_alias(tbl_alias)
        where_clause.set_alias(tbl_alias)
        group_clause.set_alias(tbl_alias)
        having_clause.set_alias(tbl_alias)
        order_clause.set_alias(tbl_alias)

        return [sel_clause, from_clause, where_clause, group_clause, having_clause, order_clause,
                intersect_clause, union_clause, except_clause, db]

    @staticmethod
    def __parse_col_unit(unit):
        unit_op, [agg_id, col_id, is_dist], col_b = unit
        agg = KeywordStr.get_agg(agg_id)
        return col_id, agg, is_dist
