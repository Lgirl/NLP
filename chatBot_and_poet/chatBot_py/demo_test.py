# -*- coding：utf-8 -*-
# -*- author：zzZ_CMing  CSDN address:https://blog.csdn.net/zzZ_CMing
# -*- 2018/07/31；14:23
# -*- python3.5
import random

import jieba
import numpy as np
import tensorflow as tf
from flask import request
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq

from chatBot_py import word_token

size = 8               # LSTM神经元size
GO_ID = 1              # 输出序列起始标记
EOS_ID = 2             # 结尾标记
PAD_ID = 0             # 空值填充0
min_freq = 1           # 样本频率超过这个值才会存入词表
epochs = 2000          # 训练次数
batch_num = 1000       # 参与训练的问答对个数
input_seq_len = 25         # 输入序列长度
output_seq_len = 50        # 输出序列长度
init_learning_rate = 0.5     # 初始学习率
from transplate.transplate import translate,get_reuslt
# sess = tf.Session()

class chatBot():
    def __init__(self):
        self.wordToken = word_token.WordToken()

        # 放在全局的位置，为了动态算出 num_encoder_symbols 和 num_decoder_symbols
        max_token_id = self.wordToken.load_file_list(['./chatBot_train_data/question', './chatBot_train_data/answer'], min_freq)
        self.num_encoder_symbols = max_token_id + 5
        self.num_decoder_symbols = max_token_id + 5

    def get_id_list_from(self,sentence):
        """
        得到分词后的ID
        """
        sentence_id_list = []
        seg_list = jieba.cut(sentence)
        for str in seg_list:
            id = self.wordToken.word2id(str)
            if id:
                sentence_id_list.append(self.wordToken.word2id(str))
        return sentence_id_list

    def get_train_set(self):
        """
        得到训练问答集
        """
        global num_encoder_symbols, num_decoder_symbols
        train_set = []
        with open('./chatBot_train_data/question', 'r', encoding='utf-8') as question_file:
            with open('./chatBot_train_data/answer', 'r', encoding='utf-8') as answer_file:
                while True:
                    question = question_file.readline()
                    answer = answer_file.readline()
                    if question and answer:
                        # strip()方法用于移除字符串头尾的字符
                        question = question.strip()
                        answer = answer.strip()

                        # 得到分词ID
                        question_id_list = get_id_list_from(question)
                        answer_id_list = get_id_list_from(answer)
                        if len(question_id_list) > 0 and len(answer_id_list) > 0:
                            answer_id_list.append(EOS_ID)
                            train_set.append([question_id_list, answer_id_list])
                    else:
                        break
        return train_set

    def get_samples(self,train_set, batch_num):
        """
        构造样本数据:传入的train_set是处理好的问答集
        batch_num:让train_set训练集里多少问答对参与训练
        """
        raw_encoder_input = []
        raw_decoder_input = []
        if batch_num >= len(train_set):
            batch_train_set = train_set
        else:
            random_start = random.randint(0, len(train_set) - batch_num)
            batch_train_set = train_set[random_start:random_start + batch_num]

        # 添加起始标记、结束填充
        for sample in batch_train_set:
            raw_encoder_input.append([PAD_ID] * (input_seq_len - len(sample[0])) + sample[0])
            raw_decoder_input.append([GO_ID] + sample[1] + [PAD_ID] * (output_seq_len - len(sample[1]) - 1))

        encoder_inputs = []
        decoder_inputs = []
        target_weights = []

        for length_idx in range(input_seq_len):
            encoder_inputs.append(
                np.array([encoder_input[length_idx] for encoder_input in raw_encoder_input], dtype=np.int32))
        for length_idx in range(output_seq_len):
            decoder_inputs.append(
                np.array([decoder_input[length_idx] for decoder_input in raw_decoder_input], dtype=np.int32))
            target_weights.append(np.array([
                                               0.0 if length_idx == output_seq_len - 1 or decoder_input[
                                                                                              length_idx] == PAD_ID else 1.0
                                               for decoder_input in raw_decoder_input
                                               ], dtype=np.float32))
        return encoder_inputs, decoder_inputs, target_weights

    def seq_to_encoder(self,input_seq):
        """
        从输入空格分隔的数字id串，转成预测用的encoder、decoder、target_weight等
        """
        input_seq_array = [int(v) for v in input_seq.split()]
        encoder_input = [PAD_ID] * (input_seq_len - len(input_seq_array)) + input_seq_array
        decoder_input = [GO_ID] + [PAD_ID] * (output_seq_len - 1)
        encoder_inputs = [np.array([v], dtype=np.int32) for v in encoder_input]
        decoder_inputs = [np.array([v], dtype=np.int32) for v in decoder_input]
        target_weights = [np.array([1.0], dtype=np.float32)] * output_seq_len
        return encoder_inputs, decoder_inputs, target_weights

    def get_model(self,feed_previous=False):
        """
        构造模型
        """
        learning_rate = tf.Variable(float(init_learning_rate), trainable=False, dtype=tf.float32)
        learning_rate_decay_op = learning_rate.assign(learning_rate * 0.9)

        encoder_inputs = []
        decoder_inputs = []
        target_weights = []
        for i in range(input_seq_len):
            encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
        for i in range(output_seq_len + 1):
            decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
        for i in range(output_seq_len):
            target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

        # decoder_inputs左移一个时序作为targets
        targets = [decoder_inputs[i + 1] for i in range(output_seq_len)]

        cell = tf.contrib.rnn.BasicLSTMCell(size)

        # 这里输出的状态我们不需要
        outputs, _ = seq2seq.embedding_attention_seq2seq(
            encoder_inputs,
            decoder_inputs[:output_seq_len],
            cell,
            num_encoder_symbols=self.num_encoder_symbols,
            num_decoder_symbols=self.num_decoder_symbols,
            embedding_size=size,
            output_projection=None,
            feed_previous=feed_previous,
            dtype=tf.float32)

        # 计算加权交叉熵损失
        loss = seq2seq.sequence_loss(outputs, targets, target_weights)
        # 梯度下降优化器
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        # 优化目标：让loss最小化
        update = opt.apply_gradients(opt.compute_gradients(loss))
        # 模型持久化
        saver = tf.train.Saver(tf.global_variables())

        return encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate

    def train(self):
        """
        训练过程
        """
        train_set = get_train_set()
        with tf.Session() as sess:
            encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate = get_model()
            sess.run(tf.global_variables_initializer())

            # 训练很多次迭代，每隔100次打印一次loss，可以看情况直接ctrl+c停止
            previous_losses = []
            for step in range(epochs):
                sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = get_samples(train_set, batch_num)
                input_feed = {}
                for l in range(input_seq_len):
                    input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
                for l in range(output_seq_len):
                    input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
                    input_feed[target_weights[l].name] = sample_target_weights[l]
                input_feed[decoder_inputs[output_seq_len].name] = np.zeros([len(sample_decoder_inputs[0])],
                                                                           dtype=np.int32)
                [loss_ret, _] = sess.run([loss, update], input_feed)
                if step % 100 == 0:
                    print('step=', step, 'loss=', loss_ret, 'learning_rate=', learning_rate.eval())
                    # print('333', previous_losses[-5:])

                    if len(previous_losses) > 5 and loss_ret > max(previous_losses[-5:]):
                        sess.run(learning_rate_decay_op)
                    previous_losses.append(loss_ret)

                    # 模型参数保存
                    saver.save(sess, './output_chat/' + str(epochs) + '/demo_')
                    # saver.save(sess, './output_chat/' + str(epochs) + '/demo_' + step)

    def infer(self):
        encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate = self.get_model(
            feed_previous=True)
        saver.restore(sess, './output_chat/' + str(epochs) + '/demo_')
        return encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate
        # self.predict(encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate)

    def predict(self,sess,encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate):
        """
        预测过程
        """
        # encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate = self.get_model(
        #     feed_previous=True)
        # saver.restore(sess, './output_chat/' + str(epochs) + '/demo_')
        # sys.stdout.write("you ask>> ")
        # sys.stdout.flush()
        # input_seq = sys.stdin.readline()
        input_seq = request.args.get('question')
        input_seq = input_seq.strip()
        input_seq = input_seq.strip()
        if "e" in input_seq:
            input_seq = translate(input_seq)
        input_id_list = self.get_id_list_from(input_seq)
        if (len(input_id_list)):
            sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = self.seq_to_encoder(
                ' '.join([str(v) for v in input_id_list]))

            input_feed = {}
            for l in range(input_seq_len):
                input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
            for l in range(output_seq_len):
                input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
                input_feed[target_weights[l].name] = sample_target_weights[l]
            input_feed[decoder_inputs[output_seq_len].name] = np.zeros([2], dtype=np.int32)

            # 预测输出
            outputs_seq = sess.run(outputs, input_feed)
            # 因为输出数据每一个是num_decoder_symbols维的，因此找到数值最大的那个就是预测的id，就是这里的argmax函数的功能
            outputs_seq = [int(np.argmax(logit[0], axis=0)) for logit in outputs_seq]
            # 如果是结尾符，那么后面的语句就不输出了
            if EOS_ID in outputs_seq:
                outputs_seq = outputs_seq[:outputs_seq.index(EOS_ID)]
            outputs_seq = [self.wordToken.id2word(v) for v in outputs_seq]
            return " ".join(outputs_seq)
        else:
            # print("WARN：词汇不在服务区")
            return "听不懂你在讲什么啦！"
            # sys.stdout.write("you ask>>")
            # sys.stdout.flush()
            # input_seq = sys.stdin.readline()


if __name__ == "__main__":
    if "trains" == 'train':
        train()
    else:
        chatBot = chatBot()
        chatBot.infer()
