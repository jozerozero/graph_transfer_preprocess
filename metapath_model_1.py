import tensorflow as tf
from tensorflow.contrib import rnn
import tensorboard_logger
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from argparse import ArgumentParser
import os
import datetime

from utils.data_loader import data_loader, valid_data_loader, get_input_metapath_instance_list


class BaseMetapathModel:

    def __init__(self, user_num, item_num, embedding_size=512, lstm_layer_num=3,
                 lstm_hidden_state_size=512, user_metapaht_len=3, loss_type="classification",
                 mlp_hidden_size_list="1024,512,200,5"):
        self.user_num = user_num
        self.item_num = item_num
        self.lstm_layer_num = lstm_layer_num
        self.lstm_hidden_state_size = lstm_hidden_state_size
        self.user_metapath_len = user_metapaht_len
        self.embedding_size = embedding_size
        self.mlp_hidden_size_list = map(int, mlp_hidden_size_list.split(","))
        self.saver = None

    def forward(self, user_metapath, item_metapath, is_training, rnn_cell=rnn.LSTMCell):

        user_item_embedding_table = \
            tf.get_variable(name="user_embedding", shape=[self.user_num+self.item_num, self.embedding_size],
                            initializer=tf.contrib.layers.xavier_initializer())

        user_metapath_input = tf.nn.embedding_lookup(user_item_embedding_table, user_metapath)

        user_forward_rnn_cell_list = [rnn_cell(num_units=self.lstm_hidden_state_size, name="user_fw_%d" % idx)
                                      for idx in range(self.lstm_layer_num)]
        user_backward_rnn_cell_list = [rnn_cell(num_units=self.lstm_hidden_state_size, name="user_bw_%d" % idx)
                                       for idx in range(self.lstm_layer_num)]
        user_multilayer_forward_rnn_cell = rnn.MultiRNNCell(user_forward_rnn_cell_list)
        user_multilayer_backward_rnn_cell = rnn.MultiRNNCell(user_backward_rnn_cell_list)

        user_output, _, _ = rnn.static_bidirectional_rnn(cell_fw=user_multilayer_forward_rnn_cell,
                                                         cell_bw=user_multilayer_backward_rnn_cell,
                                                         inputs=tf.unstack(user_metapath_input, num=3, axis=1),
                                                         dtype=tf.float32)

        item_metapath_input = tf.nn.embedding_lookup(user_item_embedding_table, item_metapath)
        item_forward_rnn_cell_list = [rnn_cell(num_units=self.lstm_hidden_state_size, name="item_fw_%d" % idx)
                                      for idx in range(self.lstm_layer_num)]
        item_backward_rnn_cell_list = [rnn_cell(num_units=self.lstm_hidden_state_size, name="item_bw_%d" % idx)
                                       for idx in range(self.lstm_layer_num)]
        item_multilayer_forward_rnn_cell = rnn.MultiRNNCell(item_forward_rnn_cell_list)
        item_multilayer_backward_rnn_cell = rnn.MultiRNNCell(item_backward_rnn_cell_list)

        item_output, _, _ = rnn.static_bidirectional_rnn(cell_fw=item_multilayer_forward_rnn_cell,
                                                         cell_bw=item_multilayer_backward_rnn_cell,
                                                         inputs=tf.unstack(item_metapath_input, num=3, axis=1),
                                                         dtype=tf.float32)

        output = tf.concat([user_output[0], item_output[0]], axis=1)
        for mlp_num_unit in self.mlp_hidden_size_list:
            output = tf.keras.layers.Dense(units=mlp_num_unit, activation=tf.nn.relu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())(output)
            output = tf.layers.dropout(output, training=is_training)

        self.saver = tf.train.Saver(max_to_keep=100)

        return output


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=1024, help="batch size", type=int)
    parser.add_argument("--user_num", default=24419, help="total user number", type=int)
    parser.add_argument("--item_num", default=27810, help="total item number", type=int)
    parser.add_argument("--training_step", default=1000000, help="total training step", type=int)
    parser.add_argument("--prefix", default="methpath_baseline1", help="model name", type=str)
    parser.add_argument("--save_model_path", default="model_result", help="save model path", type=str)
    parser.add_argument("--learning_rate", default=0.001, help="learning rate", type=float)
    parser.add_argument("--evaluate_steps", default=500, help="evaluate per steps", type=int)
    args = parser.parse_args()

    model_id = "lr_%s-" % args.learning_rate + datetime.datetime.now().strftime("%Y-%m-%d-%X").replace(":", "-")
    model_path = os.path.join(args.save_model_path, args.prefix, model_id)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    tensorboard_logger.configure(model_path)

    user_metapath = tf.placeholder(name="user_metapath", shape=[None, 3], dtype=tf.int32)
    item_metapath = tf.placeholder(name="item_metapath", shape=[None, 3], dtype=tf.int32)
    label = tf.placeholder(name="label", shape=[None, 5], dtype=tf.int32)
    is_training = tf.placeholder(name="train_mode", shape=[], dtype=tf.bool)

    model = BaseMetapathModel(user_num=args.user_num, item_num=args.item_num)
    output = model.forward(user_metapath=user_metapath, item_metapath=item_metapath, is_training=is_training)

    greg_loss = 5e-5 * tf.reduce_mean([tf.nn.l2_loss(x) for x in tf.trainable_variables()])
    label_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=label))
    total_loss = label_loss + greg_loss
    total_loss = label_loss
    train_option = tf.train.AdamOptimizer(args.learning_rate).minimize(total_loss)

    all_meta_path = get_input_metapath_instance_list()
    train_iterator = data_loader(input_metapath_instance_list=all_meta_path, mode="train", batch_size=args.batch_size)
    # valid_iterator = valid_data_loader(input_metapath_instance_list=all_meta_path, mode="test",
    #                                    batch_size=args.batch_size)
    # test_iterator = valid_data_loader(input_metapath_instance_list=all_meta_path, mode="val",
    #                                   batch_size=args.batch_size)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        train_loss_list = list()
        for step in range(args.training_step):

            train_batch_user_metapaths, train_batch_item_metapaths, train_batch_labels = next(train_iterator)

            _, train_batch_loss = session.run([train_option, total_loss],
                                              feed_dict={user_metapath: train_batch_user_metapaths,
                                                         item_metapath: train_batch_item_metapaths,
                                                         label: train_batch_labels, is_training: True})
            train_loss_list.append(train_batch_loss)
            # print(train_batch_loss)

            if step % args.evaluate_steps == 0 and step != 0:
                train_loss = sum(train_loss_list) / float(args.evaluate_steps)
                train_loss_list = list()
                tensorboard_logger.log_value(name="train_loss", value=train_loss, step=step)

                valid_iterator = valid_data_loader(input_metapath_instance_list=all_meta_path, mode="test",
                                                   batch_size=args.batch_size)
                test_iterator = valid_data_loader(input_metapath_instance_list=all_meta_path, mode="val",
                                                  batch_size=args.batch_size)
                val_loss_list = list()
                is_end = False
                val_steps = 0
                while not is_end:
                    val_batch_user_metapaths, val_batch_item_metapaths, val_batch_labels, is_end = next(valid_iterator)

                    if len(val_batch_labels) == 0:
                        break

                    val_batch_loss = session.run(total_loss, feed_dict={user_metapath: val_batch_user_metapaths,
                                                                        item_metapath: val_batch_item_metapaths,
                                                                        label: val_batch_labels, is_training: False})
                    val_loss_list.append(val_batch_loss)
                    val_steps += 1
                    if val_steps == 10:
                        break

                val_loss = sum(val_loss_list) / float(val_steps)
                tensorboard_logger.log_value(name="val_loss", value=val_loss)

                test_loss_list = list()
                is_end = False
                test_steps = 0
                while not is_end:
                    test_batch_user_metapaths, test_batch_item_metapaths, test_batch_labels, is_end = \
                        next(valid_iterator)

                    if len(test_batch_labels) == 0:
                        break

                    test_batch_loss = session.run(total_loss, feed_dict={user_metapath: test_batch_user_metapaths,
                                                                         item_metapath: test_batch_item_metapaths,
                                                                         label: test_batch_labels, is_training: False})
                    test_loss_list.append(test_batch_loss)
                    test_steps += 1
                    if test_steps == 10:
                        break
                test_loss = sum(test_loss_list) / float(test_steps)
                print("steps:%d\ttrain loss:%f\tval loss:%f\ttest loss:%f" % (step, train_loss, val_loss, test_loss))
                model.saver.save(session, os.path.join(model_path, "%d-model.ckpt"%step))