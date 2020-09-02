from util import SpiderDataLoader
from util import format_bert_input
from util import format_sample_t3
from trainer import Trainer
from trainer import TrainingConfig
from trainer import TrainingState
from graph_net import GraphNet
from sql import SqlKeyWords
from sql import SqlQueryLoader
import warnings
warnings.simplefilter('ignore')


# Origin File
sql_query_trn, sql_query_dev = SpiderDataLoader.load_spider_query()
db_ids_trn, db_ids_dev = SpiderDataLoader.load_spider_sql_db_id()
sql_json_trn, sql_json_dev = SpiderDataLoader.load_spider_sql_json()
# Annotation File
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

bert_inputs_trn = format_bert_input(lemma_trn, db_ids_trn, databases)
kw_inputs_trn, kw_ins_trn, kw_label_trn = format_sample_t3(sql_json_trn, db_ids_trn, databases, bert_inputs_trn)

bert_inputs_dev = format_bert_input(lemma_dev, db_ids_dev, databases)
kw_inputs_dev, kw_ins_dev, kw_label_dev = format_sample_t3(sql_json_dev, db_ids_dev, databases, bert_inputs_dev)
print("Training Data Ready")

model = GraphNet()
trainer = Trainer(model, gpu_num=2)

# Training Detail
training_config = TrainingConfig()
training_state = TrainingState(task_num=1)

input_list_trn = []
truth_list_trn = []
input_list_dev = [*bert_inputs_dev]
truth_list_dev = []

pad_tok = SqlKeyWords.get_index(SqlKeyWords.END_TOK)
input_pad = [0, 0, 0, 0, -1, 0, 0, 0]
truth_pad = [0]

training_config.set_batch_size(256)
training_config.set_epoch(200)
training_config.set_input_trn(input_list_trn)
training_config.set_truth_trn(truth_list_trn)
training_config.set_input_pad(input_pad)
training_config.set_truth_pad(truth_pad)
training_config.set_input_dev(input_list_dev)
training_config.set_truth_dev(truth_list_dev)
training_config.set_prediction_func(trainer.model.child_link)
training_config.set_session_name("kp")

training_config.add_loss_func(trainer.loss_pool.calc_bce_loss)
training_config.add_eval_func(trainer.metrics.bi_cls_metric)

# Configure Trainer
trainer.training_state = training_state
trainer.training_config = training_config
trainer.train()
