from util import SpiderDataLoader
from util import format_sample_t1
from graph_net import GraphNet
from trainer import Trainer
from trainer import TrainingConfig
from trainer import TrainingState
from sql import SqlKeyWords
import warnings
warnings.simplefilter('ignore')


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

model = GraphNet()
#trainer = Trainer(model, gpu_num=3)
trainer = Trainer(model)
samples_trn = format_sample_t1(lemma_trn, sql_json_trn, db_ids_trn, slot_trn, databases)
samples_dev = format_sample_t1(lemma_dev, sql_json_dev, db_ids_dev, slot_dev, databases)
print("Training Data Ready")

# Training Detail
training_config = TrainingConfig()
training_state = TrainingState(task_num=1)

input_list_trn = [*samples_trn["bert_input"]]
truth_list_trn = [*samples_trn["slot"]]
input_list_dev = [*samples_dev["bert_input"]]
truth_list_dev = [*samples_dev["slot"]]
input_pad = [0, 0, 0, 0, -1]
truth_pad = [0]

training_config.set_batch_size(32)
training_config.set_epoch(1)
training_config.set_input_trn(input_list_trn)
training_config.set_truth_trn(truth_list_trn)
training_config.set_input_pad(input_pad)
training_config.set_truth_pad(truth_pad)
training_config.set_input_dev(input_list_dev)
training_config.set_truth_dev(truth_list_dev)
training_config.set_prediction_func(trainer.model.entity_linking)
training_config.set_session_name("sf")

training_config.add_loss_func(trainer.loss_pool.calc_mce_loss)
training_config.add_eval_func(trainer.metrics.mul_cls_metric)

# Configure Trainer
trainer.training_state = training_state
trainer.training_config = training_config
trainer.train()
