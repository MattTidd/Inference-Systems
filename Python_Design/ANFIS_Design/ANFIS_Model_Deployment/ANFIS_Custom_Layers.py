"""
This file hosts the custom layers and constraints used in the ANFIS model. 

It is meant to be imported when the ANFIS model is deployed. 

"""
#################################### Import Packages: ####################################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, constraints
from keras.layers import Layer
from itertools import product

#################################### Define  Classes: ####################################

# need to define a constraint for training the parameters:
class OrderedConstraint(constraints.Constraint):
    # constructor:
    def __init__(self):
        pass

    # call function for constraint:
    def __call__(self, W):
        return tf.sort(W, axis = 2)

# first layer -> membership layer:
class MF_Layer(Layer): 
    # constructor:
    def __init__(self, num_inputs, num_mfs, mf_type, **kwargs):
        super(MF_Layer, self).__init__(**kwargs)
        self.num_inputs = num_inputs
        self.num_mfs = num_mfs

        # check if string passed:
        if not type(mf_type) is str:
            raise TypeError('Only strings are permitted to be passed')
        
        # check if a recognized membership function was passed:
        if any(mf in mf_type for mf in ['Smoothed Triangular', 'Gaussian', 'Generalized Bell']):
            pass
        else:
            raise ValueError('Unrecognized MF passed to function')
        
        # assign mf type to object, which will determine the number of parameters generated:
        if mf_type == 'Gaussian':
            self.mf_type = 'Gaussian'
            self.num_antecedents = 2
            self.constraints = None
            self.init_max = 50.0
            self.init_min = 0.0
        elif mf_type == 'Smoothed Triangular':
            self.mf_type = 'Smoothed Triangular'
            self.num_antecedents = 3
            self.constraints = OrderedConstraint()
            self.init_max = 50.0
            self.init_min = 0.0
        elif mf_type == 'Generalized Bell':
            self.mf_type = 'Generalized Bell'
            self.num_antecedents = 3
            self.constraints = None
            self.init_max = 50.0
            self.init_min = 1.0

        # need to initialize antecedent parameters
        self.mf_params = self.add_weight(
            shape = (self.num_inputs, self.num_mfs, self.num_antecedents),             
            initializer= tf.keras.initializers.RandomUniform(self.init_min, self.init_max),
            trainable = True,
            name = 'Antecedent_Params',
            constraint = self.constraints
        )

    # custom setting of weights:
    def set_weights(self, params):
        # this function is used to set weights based on what a user provides
        # user must provide weights in the form of a np.array of shape (num_mfs, num_params)

        if params.shape != (self.num_inputs, self.num_mfs, self.num_antecedents):
            raise ValueError(f'Parameters provided are not of correct shape, expected ({self.num_inputs}, {self.num_mfs}, {self.num_antecedents})')

        self.mf_params = params

    # function call:
    def call(self, inputs):
        # need to initialize the membership values:
        membership_values = []

        # for every input:
        for i in range(self.num_inputs):
            # get the memberships for that input:
            input_mf_params = self.mf_params[i]

            # need to now compute the fuzzified value for each membership function:
            fuzzified_values = []

            # for every membership function:
            for j in range(self.num_mfs):

                # if gaussian:
                if self.mf_type == 'Gaussian':
                    # define parameters:
                    mean = input_mf_params[j, 0]  # mean of the gaussian
                    std = input_mf_params[j, 1]   # standard deviation of the gaussian

                    # compute output:
                    output = tf.exp(-0.5 * tf.square((inputs[:, i] - mean) / (std + 1e-6)))
                    fuzzified_values.append(output)

                # if smoothed triangular:
                if self.mf_type == 'Smoothed Triangular':
                    # define parameters
                    a = input_mf_params[j, 0]   # a parameter
                    b = input_mf_params[j, 1]   # b parameter
                    c = input_mf_params[j, 2]   # c parameter

                    # smoothing factor beta:
                    beta = 100.0

                    # check if we are on the edges:
                    is_left_edge = tf.equal(a, b)
                    is_right_edge = tf.equal(b, c)

                    # compute softplus-based smoothed triangular membership function:
                    left = tf.nn.softplus(beta * (inputs[:, i] - a)) / (tf.nn.softplus(beta * (b - a)) + 1e-6)
                    right = tf.nn.softplus(beta * (c - inputs[:, i])) / (tf.nn.softplus(beta * (c - b)) + 1e-6)

                    # deal with edge case:
                    left = tf.where((inputs[:, i] == a) & is_left_edge, 1.0, left)
                    right = tf.where((inputs[:, i] == c) & is_right_edge, 1.0, right)

                    # compute output:
                    output = tf.maximum(0.0, tf.minimum(left, right))
                    fuzzified_values.append(output)

                # if generalized bell:
                if self.mf_type == 'Generalized Bell':
                    # define parameters
                    a = input_mf_params[j, 0]
                    b = input_mf_params[j, 1]
                    c = input_mf_params[j, 2]

                    # clamp b:
                    b = tf.clip_by_value(b, 1e-6, 5.0)

                    # compute output:
                    output = 1 / (1 + tf.abs((inputs[:, i] - c) / (a + 1e-6)) ** (2 * b))
                    fuzzified_values.append(output)
            
            # need to now stack the mf values for that given input:
            membership_values.append(tf.stack(fuzzified_values, axis = -1))

        # stack everything and return:
        return tf.stack(membership_values, axis = 1)
    
# second layer -> firing strength layer:
class FS_Layer(Layer):
    # constructor:
    def __init__(self, num_inputs, num_mfs, **kwargs):
        super(FS_Layer, self).__init__(**kwargs)
        self.num_inputs = num_inputs
        self.num_mfs = num_mfs
        self.num_rules = num_mfs ** num_inputs

    # call function:
    def call(self, membership_values):
        # this layer accepts the membership values, which have shape (batch_size, num_inputs, num_mfs):
        batch_size = tf.shape(membership_values)[0]

        # initialize the firing strengths:
        firing_strengths = tf.ones((batch_size, self.num_rules), dtype = tf.float32)

        # generate all the rule combinations:
        rules = list(product(range(self.num_mfs), repeat = self.num_inputs))    # example [(0, 0, 0), (0, 0, 1), ...]

        # need to check each input, each mf combination, and multiply their values together:
        for rule_index, combination in enumerate(rules):
            # print(f'combination: {combination}')
            rule_strength = tf.ones((batch_size, ), dtype = tf.float32)

            # for every input and membership function:
            for input_index, mf_index in enumerate(combination):
                # print(f'input: {input_index + 1} | mf: {mf_index + 1}')

                # correctly extract the fuzzified values based on the combination index:
                rule_strength *= membership_values[:, input_index, mf_index] + 1e-6
            
            # update the firing strengths:
            rule_strength = tf.expand_dims(rule_strength, axis = -1)  # shape: (batch_size, 1)
            firing_strengths = tf.concat(
                [firing_strengths[:, :rule_index], rule_strength, firing_strengths[:, rule_index + 1:]],
                axis = 1,
            )
            # print(f'firing strength: {firing_strengths}')

        return firing_strengths
    
# third layer -> normalization layer:
class NM_Layer(Layer):
    # constructor:
    def __init__(self, num_inputs, num_mfs, **kwargs):
        super(NM_Layer, self).__init__(**kwargs)
        self.num_inputs = num_inputs
        self.num_mfs = num_mfs

    # call function:
    def call(self, firing_strengths):
        # this function accepts inputs of size (batch_size, num_rules).
        # need to first get the total firing strength:
        total_firing_strength = tf.reduce_sum(firing_strengths, axis = 1, keepdims = True)
        
        # can now normalize the firing strengths:
        normalized_strengths = firing_strengths / (total_firing_strength + 1e-10)   # add a buffer in case the total firing strength is zero

        return normalized_strengths
    
# fourth layer -> consequent layer:
class CN_Layer(Layer):
    # constructor: 
    def __init__(self, num_inputs, num_mfs, **kwargs):
        super(CN_Layer, self).__init__(**kwargs)
        self.num_inputs = num_inputs
        self.num_mfs = num_mfs
        self.num_rules = num_mfs ** num_inputs

        # need to initialize the consequent parameters:
        self.consequent_params = self.add_weight(
            shape = (self.num_rules, self.num_inputs + 1),
            initializer = tf.keras.initializers.RandomUniform(-1.0, 1.0, seed = 1234),
            trainable = True,
            name = 'Consequent_Params'
        )

    # this function is used for manually setting the consequent parameters:
    def set_cons(self, params):
        # this function accepts parameters as an array of size (num_rules, num_inputs + 1):
        if params.shape != (self.num_rules, self.num_inputs + 1):
            raise ValueError(f'Parameters provided are not of correct shape, expected ({self.num_rules}, {self.num_inputs + 1})')
        
        # assign parameters:
        self.consequent_params = params

    # call function:
    def call(self, input_list):
        # unpack inputs from list:
        normalized_strengths, inputs = input_list

        # get the batch size:
        batch_size = tf.shape(normalized_strengths)[0]

        # the output is given by the multiplication of the inputs with the consequent weights,
        # such as: o_k = w_bar_k * (x_1 * p_k + x_2 * q_k + x_3 * r_k + ... + s_k)
        # can therefore extend the inputs to be (batch_size, num_inputs + bias) for ease of multiplication:
        inputs_with_bias = tf.concat([inputs, tf.ones((batch_size, 1), dtype = tf.float32)], axis = -1)

        # need to now reshape the normalized strengths to be of size (batch_size, num_rules, 1)
        # this effectively flips it into a 'column vector' of sorts, where each individual value is now vertically aligned
        normalized_strengths = tf.reshape(normalized_strengths, (batch_size, self.num_rules, 1))

        # get the consequent parameters, which have shape (num_rules, num_inputs + 1):
        consequent_params = self.consequent_params

        # expand inputs with bias to match the rule axis: (batch_size, num_rules, num_inputs + 1)
        inputs_with_bias_expanded = tf.expand_dims(inputs_with_bias, axis = 1)

        # calculate the consequent for each rule
        consequents = tf.reduce_sum(normalized_strengths * inputs_with_bias_expanded * consequent_params, axis = 2)

        return consequents

# fifth layer -> output layer:
class O_Layer(Layer):
    # constructor:
    def __init__(self, num_inputs, num_mfs, **kwargs):
        super(O_Layer, self).__init__(**kwargs)
        self.num_inputs = num_inputs
        self.num_output = num_mfs

    # call function:
    def call(self, consequents):
        output = tf.reduce_sum(consequents, axis = 1, keepdims = True)
        return output
