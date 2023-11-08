from layers import *
from metrics import *
import os
flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        print('Model  do not have _build Implemention')
        raise NotImplementedError

    def build(self):
        # print('Model_build_function')
        """ Wrapper for _build() """        
        with tf.variable_scope(self.name):
            self._build()
        
        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]
        

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()
        self._dice()
        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def _dice(self):
        raise NotImplementedError

    def save(self, sess=None,path=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        if not path:
            path = "tmp/%s.ckpt" % self.name
        else:
            path = path +"%s.ckpt" % self.name
        save_path = saver.save(sess, path)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None, path=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        if path:
            save_path = os.path.join(path, self.name+'.ckpt')
            print('loading path', save_path)
        else:
            save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)



class GCN(Model):
    def __init__(self, placeholders, input_dim, layer_num, **kwargs):
        super(GCN, self).__init__(**kwargs)
        # print('_____GCN_init____')
        self.layer_num = layer_num
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        print('learning rate:', FLAGS.learning_rate)
        # self.edge_mask = 1 - self.placeholders['labels_mask']
        self.build()


    def _loss(self):
        # Weight decay loss
        for num in range(self.layer_num):
            for var in self.layers[num].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        
        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])
        

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])


    def _dice(self):
        self.dice = multiclass_dice(self.outputs, self.placeholders['labels'])


    def _build(self):
        print('GCN_build_function')
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=False,
                                            sparse_inputs=True,
                                            logging=self.logging))

        for i in range(self.layer_num):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=False,
                                                logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act= lambda x: x,
                                            dropout=False,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)
