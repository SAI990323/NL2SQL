from util import SpiderDataLoader
from agent import interact
from sql import SqlQueryLoader
from graph import SqlGraphConverter

# Data Loading
sql_query_trn, sql_query_dev = SpiderDataLoader.load_spider_query()
db_ids_trn, db_ids_dev = SpiderDataLoader.load_spider_sql_db_id()
sql_json_trn, sql_json_dev = SpiderDataLoader.load_spider_sql_json()
slot_trn, slot_dev = SpiderDataLoader.load_slot_filling()
lemma_trn, lemma_dev = SpiderDataLoader.load_lemma()
databases = SpiderDataLoader.load_spider_dbs()

# Illegal index
delete_idx = [6054, 6053, 6052, 6051, 3153, 526, 525, 524, 523, 520, 519]
for i in delete_idx:
    sql_query_trn.pop(i)
    db_ids_trn.pop(i)
    sql_json_trn.pop(i)
print("Data Loaded")

data_size_trn = len(sql_json_trn)
for i in range(data_size_trn):
    sql = SqlQueryLoader.load(sql_json_trn[i], databases[db_ids_trn[i]])
    t_1 = SqlGraphConverter.sql2graph(sql)
    #t = interact(sql)
    print(sql)
    #t_1.show()
    #t.show()
    print("-----------")
print(data_size_trn)


