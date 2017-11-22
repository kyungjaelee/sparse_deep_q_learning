import tensorflow as tf
import numpy as np

class Network:
    def __init__(self,input_dim,output_dim,hidden_spec,scope_name='Network',learning_rate=1e-4,seed=0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_spec = len(hidden_spec)
        self.hidden_spec = hidden_spec
        self.scope_name = scope_name
        self.learning_rate = learning_rate
        self.seed=seed
        
        self.rho, self.inputs, self.outputs, self.weights, self.biases, self.weights_old, self.biases_old = self._generate_tf_variables(hidden_spec)
        self.hidden_layers, self.network_outputs, self.hidden_layers_old, self.network_outputs_old = self._build_hidden(hidden_spec)        
        self.loss, self.optim, self.errors = self._build_loss(learning_rate)
        self.update_ops = self._build_update_old_params()
        
    def _generate_tf_variables(self,hidden_spec):
        with tf.variable_scope(self.scope_name):
            rho = tf.placeholder(tf.float32,shape=[None, self.output_dim], name="batch_weights")
            inputs = tf.placeholder(tf.float32,shape=[None, self.input_dim],name='inputs')
            outputs = tf.placeholder(tf.float32,shape=[None, self.output_dim],name='outputs')
            
            weights = []
            biases = []
            weights_old = []
            biases_old = []
            prev_hidden_dim = self.input_dim
            for i in range(self.n_hidden_spec):
                ith_hidden_spec = hidden_spec[i]

                next_hidden_dim = ith_hidden_spec['dim']
                weights.append(
                    tf.get_variable(
                        name='W'+str(i),
                        shape=[prev_hidden_dim,next_hidden_dim],
                        initializer=tf.random_normal_initializer(mean=0.0,stddev=0.05,seed=self.seed)
                    )
                )
                biases.append(
                    tf.get_variable(
                        name='b'+str(i),
                        shape=[next_hidden_dim],
                        initializer=tf.constant_initializer(0.0)
                    )
                )
                weights_old.append(
                    tf.get_variable(
                        name='W_old'+str(i),
                        shape=[prev_hidden_dim,next_hidden_dim],
                        initializer=tf.random_normal_initializer(mean=0.0,stddev=0.05,seed=self.seed)
                    )
                )
                biases_old.append(
                    tf.get_variable(
                        name='b_old'+str(i),
                        shape=[next_hidden_dim],
                        initializer=tf.constant_initializer(0.0)
                    )
                )
                prev_hidden_dim = next_hidden_dim
            
            weights.append(
                tf.get_variable(
                    name='W'+str(self.n_hidden_spec),
                    shape=[prev_hidden_dim,self.output_dim],
                    initializer=tf.random_normal_initializer(mean=0.0,stddev=0.05,seed=self.seed)
                )
            )
            biases.append(
                tf.get_variable(
                    name='b'+str(self.n_hidden_spec),
                    shape=[self.output_dim],
                    initializer=tf.constant_initializer(0.0)
                )
            )
            weights_old.append(
                tf.get_variable(
                    name='W_old'+str(self.n_hidden_spec),
                    shape=[prev_hidden_dim,self.output_dim],
                    initializer=tf.random_normal_initializer(mean=0.0,stddev=0.05,seed=self.seed)
                )
            )
            biases_old.append(
                tf.get_variable(
                    name='b_old'+str(self.n_hidden_spec),
                    shape=[self.output_dim],
                    initializer=tf.constant_initializer(0.0)
                )
            )
        return rho, inputs, outputs, weights, biases, weights_old, biases_old
    
    def _build_update_old_params(self):
        update_ops = []
        for l in range(self.n_hidden_spec+1):
            update_ops.append(self.weights_old[l].assign(self.weights[l]))
            update_ops.append(self.biases_old[l].assign(self.biases[l]))
        return update_ops
        
    def _build_hidden(self,hidden_spec):
        prev_activation = self.inputs
        after_layer = None
        hidden_layers = []
        for i in range(self.n_hidden_spec):
            ith_hidden_spec = hidden_spec[i]
            after_layer = tf.matmul(prev_activation,self.weights[i])+self.biases[i]
            prev_activation = ith_hidden_spec['activation'](after_layer)
            hidden_layers.append(prev_activation)
        network_outputs = tf.matmul(prev_activation,self.weights[-1])+self.biases[-1]
        
        prev_activation = self.inputs
        after_layer = None
        hidden_layers_old = []
        for i in range(self.n_hidden_spec):
            ith_hidden_spec = hidden_spec[i]
            after_layer = tf.matmul(prev_activation,self.weights_old[i])+self.biases_old[i]
            prev_activation = ith_hidden_spec['activation'](after_layer)
            hidden_layers.append(prev_activation)
        network_outputs_old = tf.matmul(prev_activation,self.weights_old[-1])+self.biases_old[-1]
        
        return hidden_layers, network_outputs, hidden_layers_old, network_outputs_old

    def _build_loss(self,learning_rate):
        errors = self.outputs - self.network_outputs
        loss = tf.reduce_mean(self.rho*(tf.square(errors)))
        optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        return loss, optim, errors
    
    def get_train(self,input_batch,output_batch,batch_weights):
        fetches = [self.loss, self.errors, self.optim]
        feeds = {self.inputs:input_batch,self.outputs:output_batch,self.rho:batch_weights}
        return fetches, feeds
    
    def get_predictions(self,input_batch):
        fetches = [self.network_outputs]
        feeds = {self.inputs:input_batch}
        return fetches, feeds
    
    def get_predictions_old(self,input_batch):
        fetches = [self.network_outputs_old]
        feeds = {self.inputs:input_batch}
        return fetches, feeds