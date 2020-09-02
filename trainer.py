import torch
import torch.nn as nn
from util import padding
from sklearn import metrics
import numpy as np


class TrainingConfig:
    """ Data Transfer Object """
    def __init__(self):
        # Training Setting
        self.batch_size = 128
        self.epoch = 300
        self.session_name = ""
        self.epoch_per_validation = 1

        # Data
        self.input_list_trn = None
        self.truth_list_trn = None
        self.input_list_dev = None
        self.truth_list_dev = None
        self.input_list_test = None
        self.truth_list_test = None
        self.prediction_func = None

        # Pad tokens (is necessary?)
        self.input_pad = []
        self.truth_pad = []

        # Task specific variable
        self.__loss_funcs = []
        self.__eval_funcs = []

        # Some tasks need preparation before these process
        #  and sometimes redundant
        self.loss_calc_func = None

    def set_input_trn(self, input_list):
        self.input_list_trn = input_list

    def set_truth_trn(self, truth_list):
        self.truth_list_trn = truth_list

    def set_input_dev(self, input_list):
        self.input_list_dev = input_list

    def set_truth_dev(self, truth_list):
        self.truth_list_dev = truth_list

    def set_input_test(self, input_list):
        self.input_list_test = input_list

    def set_truth_test(self, truth_list):
        self.truth_list_test = truth_list

    def set_input_pad(self, input_pad):
        self.input_pad = input_pad

    def set_truth_pad(self, truth_pad):
        self.truth_pad = truth_pad

    def set_prediction_func(self, prediction_func):
        self.prediction_func = prediction_func

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_session_name(self, session_name):
        self.session_name = session_name

    def set_loss_calc_func(self, loss_calc_func):
        self.loss_calc_func = loss_calc_func

    def add_loss_func(self, loss_func):
        self.__loss_funcs.append(loss_func)

    def add_eval_func(self, eval_func):
        self.__eval_funcs.append(eval_func)

    def get_loss_funcs(self):
        return self.__loss_funcs

    def get_eval_funcs(self):
        return self.__eval_funcs


class TrainingState:
    """ Data Transfer Object """
    def __init__(self, task_num):
        self.task_num = task_num

        # Epoch data
        self.it_no = 0
        self.total_it = 0
        self.loss_list: list = []
        self.pred_batch: list = []
        self.gold_batch: list = []
        self.current_data_size: int = 0
        self.running_loss: list = [0 for _ in range(self.task_num)]
        self.metric_batches: list = [[] for _ in range(self.task_num)]

        # Global data
        self.current_epoch_id: int = 0
        self.best_accuracy: list = [0 for _ in range(self.task_num)]
        self.best_model_path: str = ""
        self.best_global_loss: float = 1e5

    def set_pred_batch(self, pred_batch):
        self.pred_batch = pred_batch

    def set_gold_batch(self, gold_batch):
        self.gold_batch = gold_batch

    def record_metric_batch(self, metric, task_num):
        self.metric_batches[task_num].append(metric)

    def get_best_model_path(self):
        return self.best_model_path

    def clear_epoch_session(self):
        self.it_no = 0
        self.total_it = 0
        self.loss_list: list = []
        self.pred_batch: list = []
        self.gold_batch: list = []
        self.current_data_size: int = 0
        self.running_loss: list = [0 for _ in range(self.task_num)]

    def clear_infer_session(self):
        self.metric_batches: list = [[] for _ in range(self.task_num)]

    def update_epoch(self):
        for i in range(self.task_num):
            epoch_loss = self.running_loss[i] / self.current_data_size
            print('Task-{} Epoch-{} Loss: {:.4f}'.format(i, self.current_epoch_id, epoch_loss))

        self.clear_epoch_session()
        self.current_epoch_id += 1


class Metrics:
    metric_tag = ['acc', 'F1', 'R', 'P']

    @classmethod
    def bi_cls_metric(cls, pred, truth):
        truth = truth.view(-1, )
        pred = pred.view(-1, )
        assert pred.shape == truth.shape, ("Pred:", pred.shape, "Gold:", truth.shape)
        # pred = torch.sigmoid(pred)

        threshold = 0.5
        pred = (pred >= threshold)
        truth = (truth >= threshold)
        metric = cls.__calc_bi_cls_metric(truth, pred)

        return metric

    @classmethod
    def mul_cls_metric(cls, pred, truth):
        sample_num = truth.numel()
        truth = truth.view(sample_num)
        pred = pred.view(sample_num, -1)

        pred = torch.softmax(pred, dim=-1).argmax(dim=-1)
        truth = truth.type(torch.long).view(-1, )
        assert pred.shape == truth.shape, ("Error: unequal shape between pred and label: ", pred.shape, truth.shape)

        metric = cls.__calc_mul_cls_metric(pred, truth)
        return metric

    @classmethod
    def seq_metric(cls, pred, truth):
        batch_size, seq_len, cls_size = pred.shape
        pred = torch.softmax(pred, dim=2)
        pred = pred.argmax(dim=2)

        truth = truth.reshape(batch_size, seq_len)
        pred = [(pred[i] == truth[i]).all().item() for i in range(batch_size)]
        pred = torch.LongTensor(pred)
        truth = torch.ones_like(pred)

        metric = cls.__calc_bi_cls_metric(pred, truth)
        return metric

    @classmethod
    def __calc_bi_cls_metric(cls, pred, truth):
        assert pred.shape == truth.shape, ("Right:", pred.shape, "Gold:", truth.shape)

        acc = metrics.accuracy_score(truth, pred)
        precision = metrics.precision_score(truth, pred, average='binary')
        recall = metrics.recall_score(truth, pred, average='binary')
        f1 = metrics.f1_score(truth, pred, average='binary')

        metric = {"acc": acc, "F1": f1, "R": recall, "P": precision,
                  "type": "binary"}

        return metric

    @classmethod
    def __calc_mul_cls_metric(cls, pred, truth):
        assert pred.shape == truth.shape, ("Right:", pred.shape, "Gold:", truth.shape)

        acc = metrics.accuracy_score(truth, pred, normalize=True)
        mode = "weighted"
        precision = metrics.precision_score(truth, pred, average=mode)
        recall = metrics.recall_score(truth, pred, average=mode)
        f1 = metrics.f1_score(truth, pred, average=mode)

        sub_precision = metrics.precision_score(truth, pred, average=None)
        sub_recall = metrics.recall_score(truth, pred, average=None)
        sub_f1 = metrics.f1_score(truth, pred, average=None)

        metric = {"acc": acc, "F1": f1, "R": recall, "P": precision,
                  "type": "multiple",
                  "sub_p": sub_precision,
                  "sub_r": sub_recall,
                  "sub_f1": sub_f1}

        return metric

    @classmethod
    def evaluate_mse_task(cls, pred, gold):
        gold = gold.squeeze(1)
        pred = pred.squeeze(1)

        import matplotlib.pyplot as plt
        for i, [img_p, img_g] in enumerate(zip(pred, gold)):
            img = torch.cat([img_p, img_g], dim=-1)
            img_name = "./res/%d.png" % i
            plt.imsave(img_name, img.numpy(), vmin=0, vmax=1)
        return 0


class Loss:
    def __init__(self):
        self.bce_loss = nn.BCELoss()
        self.mce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def calc_bce_loss(self, pred, gold):
        pred = pred.view(-1, )
        gold = gold.view(-1, ).float()
        loss = self.bce_loss(pred, gold)
        return loss

    def calc_mce_loss(self, pred, gold):
        sample_num = gold.numel()
        gold = gold.view(sample_num)
        pred = pred.view(sample_num, -1)
        loss = self.mce_loss(pred, gold.long())
        return loss

    def calc_mse_loss(self, pred, gold):
        loss = self.mse_loss(pred, gold)
        return loss


class Trainer:
    def __init__(self, model, gpu_num=0, **kwargs):
        super(Trainer, self).__init__(**kwargs)

        self.training_config = None
        self.training_state = None
        self.loss_pool = Loss()
        self.metrics = Metrics()

        self.model = model
        self.device = torch.device("cuda:%d" % gpu_num if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.set_device(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.model_path = "./model/"

    def train(self):
        for epoch_id in range(self.training_config.epoch):
            self.train_epoch()

            if (epoch_id+1) % self.training_config.epoch_per_validation == 0:
                self.validate()
            self.training_state.update_epoch()

    def train_epoch(self):
        batch_size = self.training_config.batch_size
        inputs = self.training_config.input_list_trn
        truths = self.training_config.truth_list_trn
        self.model.train()

        for batch_inputs, batch_truth in self.batch_generator(inputs, truths, batch_size):
            for j in range(8):
                print(batch_inputs[j][0])
            batch_pred = self.training_config.prediction_func(*batch_inputs, *batch_truth)
            self.training_state.set_pred_batch(batch_pred)
            self.training_state.set_gold_batch(batch_truth)
            self.calculate_loss()
            self.back_propagation()

    def batch_generator(self, inputs, truths, batch_size):
        total_num = len(inputs[0])
        is_rest = (total_num % batch_size) != 0
        batch_num = total_num // batch_size + int(is_rest)
        self.training_state.total_it = batch_num

        for batch_id in range(batch_num):
            self.training_state.it_no = batch_id + 1
            st = batch_id * batch_size
            end = st + batch_size

            batch_inputs = [comp[st:end] for comp in inputs]
            batch_truths = [comp[st:end] for comp in truths]
            batch_inputs = self.batch_padding(batch_inputs, self.training_config.input_pad)
            batch_truths = self.batch_padding(batch_truths, self.training_config.truth_pad)
            batch_inputs = [comp.to(self.device) for comp in batch_inputs]
            batch_truths = [comp.to(self.device) for comp in batch_truths]

            yield batch_inputs, batch_truths

    def validate(self):
        self.inference(self.training_config.input_list_dev, self.training_config.truth_list_dev)

    def test(self):
        self.inference(self.training_config.input_list_test, self.training_config.truth_list_test, save_model=False)

    def inference(self, inputs, truths, save_model=True):
        self.model.eval()
        batch_size = self.training_config.batch_size
        for batch_inputs, batch_truth in self.batch_generator(inputs, truths, batch_size):
            batch_pred = self.training_config.prediction_func(*batch_inputs, *batch_truth)
            self.training_state.set_pred_batch(batch_pred)
            self.training_state.set_gold_batch(batch_truth)
            self.calculate_metrics()

        self.evaluate(save_model)
        self.training_state.clear_infer_session()

    def evaluate(self, save_model):
        tags = ["acc", "F1", "P", "R"]
        task_metrics = []
        for batch_metrics in self.training_state.metric_batches:
            metric = {tag: 0 for tag in tags}
            data_size = len(batch_metrics)
            for tag in tags:
                for batch_metric in batch_metrics:
                    metric[tag] += batch_metric[tag]
                metric[tag] /= data_size
            task_metrics.append(metric)

        # Show result and save model
        for i in range(self.training_state.task_num):
            print("Val @ task-%d |" % i, end="")
            for tag in tags:
                v = task_metrics[i][tag]
                print("%3s: %.3f |" % (tag, v), end="")
            print()

        print("Epoch-%d validated;" % self.training_state.current_epoch_id)
        for i in range(self.training_state.task_num):
            acc = task_metrics[i]["acc"]
            model_name = self.model_path + "%s.task%d.e%d.param.best" % (self.training_config.session_name, i, self.training_state.current_epoch_id)
            if self.training_state.best_accuracy[i] < acc:
                self.training_state.best_accuracy[i] = acc
                self.training_state.best_model_path = model_name
                if save_model:
                    torch.save(self.model, model_name)
                    print("Best acc got! Model: %s saved" % model_name)
                else:
                    print("Best acc got at %s." % model_name)

    def calculate_metrics(self):
        pred_list = self.training_state.pred_batch
        gold_list = self.training_state.gold_batch
        eval_funcs = self.training_config.get_eval_funcs()

        # Evaluation
        tasks_acc = []
        for i in range(self.training_state.task_num):
            pred, gold, eval_func = pred_list[i], gold_list[i], eval_funcs[i]
            pred, gold = pred.cpu().detach(), gold.cpu().detach()
            metric = eval_func(pred, gold)
            self.training_state.record_metric_batch(metric, i)

        return tasks_acc

    def calculate_loss(self):
        pred_list = self.training_state.pred_batch
        gold_list = self.training_state.gold_batch
        loss_func_list = self.training_config.get_loss_funcs()

        loss_list = []
        for pred, gold, loss_func in zip(pred_list, gold_list, loss_func_list):
            loss = loss_func(pred, gold)
            loss_list.append(loss)
        self.training_state.loss_list = loss_list

        # print detail
        if (self.training_state.it_no % 1 == 0) or (self.training_state.it_no == self.training_state.total_it):
            print("'%s' Loss@Epoch-%d it %d/%d" % (self.training_config.session_name,
                                                   self.training_state.current_epoch_id,
                                                   self.training_state.it_no,
                                                   self.training_state.total_it), end="|")
            for i in range(self.training_state.task_num):
                print("task_%d: %.2f" % (i, self.training_state.loss_list[i].item()), end="|")
            print()

    def back_propagation(self):
        self.optimizer.zero_grad()
        loss = sum(self.training_state.loss_list)
        loss.backward()
        self.optimizer.step()
        self.training_state.current_data_size += self.training_config.batch_size

        for i in range(self.training_state.task_num):
            self.training_state.running_loss[i] += \
                self.training_state.loss_list[i].item() * self.training_config.batch_size

    @staticmethod
    def batch_padding(batch_data, pad_toks):
        assert len(batch_data) == len(pad_toks)
        batch_data_padded = []

        comp_num = len(batch_data)
        for i in range(comp_num):
            comp = batch_data[i]
            pad_tok = pad_toks[i]
            sample = comp[0]
            if type(sample) == int or type(sample) == np.int64:
                pass
            else:
                max_len = 0
                for sample in comp:
                    cur_len = len(sample)
                    if cur_len > max_len:
                        max_len = cur_len
                comp = [padding(sample, max_len, pad_tok) for sample in comp]
            batch_data_padded.append(torch.LongTensor(comp))

        return batch_data_padded
