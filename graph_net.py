import torch
import torch.nn as nn
from transformers import BertModel
from sql import SqlKeyWords


class GRUDecoder(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(GRUDecoder, self).__init__()
        self.device = None
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.relu = nn.ReLU()

        self.gru_decoder = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.output_project = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, inp, hidden):
        inp = self.relu(inp)
        output, hidden = self.gru_decoder(inp, hidden)
        projection = self.output_project(output.squeeze(1))
        return projection, output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

    def set_device(self, device):
        self.device = device


class GraphNet(nn.Module):
    def __init__(self, hidden_size=768, freeze_bert=True):
        super(GraphNet, self).__init__()
        self.device = None
        self.hidden_size = hidden_size
        self.kw_vocab_size = SqlKeyWords.kw_num

        self.node_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.node_seq_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.bert = BertModel.from_pretrained('./bert-base-uncased')

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.kw_embedding = nn.Embedding(self.kw_vocab_size, self.hidden_size)
        self.node_fc_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.node_fc_2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.context_fc = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.w_que_child = nn.Linear(self.hidden_size, self.hidden_size)

    def entity_linking(self, input_id, input_mask, segment_id, indication_id, que_tok_map, slot_truth):
        _, toks, nodes = self.bert_encode(input_id, input_mask, segment_id, indication_id, que_tok_map)
        nodes = nodes[:, 1:, :]  # remove the root node
        pred_slot = self.predict_slot(toks, nodes)
        return [pred_slot]

    def child_link(self, input_id, input_mask, segment_id, indication_id, que_tok_map, his, p_num, node, link_truth):
        hidden, tokes, nodes = self.bert_encode(input_id, input_mask, segment_id, indication_id, que_tok_map)
        pred_link = self.predict_link(tokes, nodes, his, p_num, node)
        return [pred_link]

    def bert_encode(self, input_id, input_mask, segment_id, indication_id, que_tok_map):
        output, hidden = self.bert(input_id)
        batch_size, seq_len, hidden_size = output.shape

        # Tok
        max_tok_num = (que_tok_map >= 0).sum(dim=1).max().item()  #??? >0
        que_toks = torch.zeros(batch_size, max_tok_num, hidden_size).to(self.device)

        for batch_idx in range(batch_size):
            tok_num = (que_tok_map[batch_idx] >= 0).sum()
            bert_tok_end_idx = segment_id[batch_idx].nonzero()[-1].item()
            for tok_id in range(tok_num):
                st = que_tok_map[batch_idx][tok_id]
                ed = bert_tok_end_idx if (tok_id == tok_num-1) else que_tok_map[batch_idx][tok_id+1]
                que_toks[batch_idx][tok_id] = output[batch_idx][st: ed].mean(dim=0)

        # Node
        max_node_num = indication_id.sum(dim=1).max().item()
        nodes = torch.zeros(batch_size, max_node_num, hidden_size).to(self.device)

        for batch_idx in range(batch_size):
            start_index = 1
            node_idx = 0
            node_gru, tok_repr = [], []
            for tok_idx in range(start_index, seq_len):
                if indication_id[batch_idx][tok_idx]:
                    inp = torch.stack(tok_repr, dim=0).unsqueeze(0)
                    output_en, hidden_en = self.node_encoder(inp)
                    nodes[batch_idx, node_idx] = hidden_en
                    tok_repr = []
                    node_idx += 1
                else:
                    tok_repr.append(output[batch_idx][tok_idx])

        root = torch.rand(batch_size, 1, hidden_size).to(self.device)
        nodes = torch.cat([root, nodes], dim=1)

        return hidden, que_toks, nodes

    def predict_slot(self, question_repr, node_repr):
        question_repr = self.node_fc_1(question_repr)
        node_repr = self.node_fc_2(node_repr)
        pred = torch.bmm(question_repr, node_repr.transpose(1, 2))
        return pred

    def predict_link(self, tokens, nodes, his, parents_num, child_id):
        batch_size, max_token_num, hidden_size = tokens.shape

        child_emb = torch.zeros(batch_size, hidden_size).to(self.device)
        for batch_idx in range(batch_size):
            node_idx = child_id[batch_idx]
            child_emb[batch_idx] = nodes[batch_idx][node_idx]

        parents_emb = torch.zeros(batch_size, hidden_size).to(self.device)
        for batch_idx in range(batch_size):
            num = parents_num[batch_idx]
            node_idx = his[batch_idx][:num]
            print(node_idx.shape)
            parents = nodes[batch_idx][node_idx].unsqueeze(0)
            print(parents.shape)
            print(child_emb[batch_idx].shape)
            print(child_emb[batch_idx].unsqueeze(0).shape)
            output, hidden = self.node_seq_encoder(parents)
            # fea = hidden.squeeze(0)
            # fea = output.mean(1).squeeze(0)
            fea = self.attention(output, child_emb[batch_idx].unsqueeze(0))
            parents_emb[batch_idx] = fea
            print('-------------------------------------------------------------------------------------')
        parents_emb = self.w_que_child(parents_emb)
        que_emb = torch.zeros(batch_size, hidden_size).to(self.device)
        for batch_idx in range(batch_size):
            que_emb[batch_idx] = self.attention(tokens[batch_idx].unsqueeze(0), child_emb[batch_idx].unsqueeze(0))

        context_emb = self.context_fc(torch.cat([que_emb, parents_emb], dim=1))
        pred = torch.mul(context_emb, child_emb).sum(1)
        pred = torch.sigmoid(pred)
        return pred

    def set_device(self, device):
        self.device = device

    @staticmethod
    def attention(sequence, target, bidirectional=False):
        # sequence: (B, S, H)
        # target: (B, H)
        if bidirectional:
            batch, seq_len, double_hidden_size = sequence.shape
            direction_num = 2
            direction_dim = 2
            sequence = sequence.view((batch, seq_len, direction_num, -1)).mean(dim=direction_dim, keepdim=False)

        prob = torch.bmm(sequence, target.unsqueeze(dim=-1)).squeeze(dim=-1)
        prob = torch.softmax(prob, dim=-1)
        att = (sequence * prob.unsqueeze(dim=-1)).sum(dim=1, keepdim=False)
        return att

    @staticmethod
    def batch_sequence_self_attention(batch_sequence):
        # a naive implementation of self sequence attention
        sequence_length_dim = 1
        vector_length_dim = 2
        res = torch.bmm(batch_sequence, batch_sequence.transpose(sequence_length_dim, vector_length_dim))
        return res
