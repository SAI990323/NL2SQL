
# Model directory
data_dir = "data/"
spider_dir = data_dir + "spider/"
path_to_jar = data_dir + 'model/stanford-parser-full-2018-10-17/stanford-parser.jar'
path_to_model_jar = data_dir + 'model/stanford-parser-full-2018-10-17/stanford-parser-3.9.2-models.jar'

POS = ['WRB', 'JJ', 'NNS', 'VBP', 'PRP', 'VB', '.', 'WP', 'VBZ', 'DT', 'NN', 'IN', ',', 'VBN',
       'JJS', 'TO', 'CC', 'VBG', 'NNP', 'PDT', 'CD', 'JJR', 'EX', 'VBD', 'RB', 'RBS', 'WDT',
       "''", 'UH', 'PRP$', 'WP$', '-LRB-', '-RRB-', 'POS', '``', 'NNPS', 'RBR', ':', 'RP', 'MD',
       'LS', 'FW', '$', 'SYM']
POS_PAD_TOK = "<PAD-POS>"
POS += [POS_PAD_TOK]
POS_VOCAB_SIZE = len(POS)
POS_DICT = {pos: pos_id for pos_id, pos in enumerate(POS)}

NODE_TYPES = ["col", "tab", "val"]
NODE_TYPE_PAD_TOK = "<PAD-TYPE>"
VIRTUAL_NODE_TYPE = "Virtual"
NODE_TYPES += [NODE_TYPE_PAD_TOK, VIRTUAL_NODE_TYPE]

NODE_TYPE_DICT = {n_type: type_id for type_id, n_type in enumerate(NODE_TYPES)}
NODE_TYPE_PAD_ID = NODE_TYPE_DICT[NODE_TYPE_PAD_TOK]
