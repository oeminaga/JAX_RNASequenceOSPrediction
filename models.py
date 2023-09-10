'''
This file contains the model class

Author: Okyaz Eminaga

'''
import math
import os
import pickle
from typing import Any, Tuple
import jax.numpy as jnp
from jax import value_and_grad
from jax import random
import random as rmd
import jax
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from jax.example_libraries import optimizers as jax_opt


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale *\
        random.normal(b_key, (n, ))


def random_layer(m, n, key, scale=10, init_weights=random.normal):
    '''
    Create a random layer

    Parameters
    ----------
    m : int
        Number of input neurons
    n : int
        Number of output neurons
    key : jax.random.PRNGKey
        Random key
    scale : int, optional
        Scale of the weights, by default 10
    init_weights : function, optional
        Function to initialize the weights, by default random.normal

    Returns
    -------
    jnp.array
        Random layer
    '''
    subkeys = random.split(key, 4)
    key_a, key_c, key_b, w_key = subkeys
    if init_weights == random.multivariate_normal:
        a = random.choice(key_a, jnp.array(list(range(1, 3)))) / 10.0  # 1, 10
        b = random.choice(key_b, jnp.array(list(range(1, 3)))) / 10.0  # 1, 10
        return scale * init_weights(key_c,
                                    mean=jnp.array([b]),
                                    cov=jnp.array([[a]]),
                                    shape=(n, m))[..., 0]
    if init_weights == random.pareto:
        b = random.choice(key_b,
                          jnp.array(list(range(50, 52))))  # 50, 100
        return scale * init_weights(key_c, b=b, shape=(n, m))
    if init_weights == random.poisson:
        # lam = random.choice(key_b, jnp.array(list(range(1, 3))))  # 1, 10
        lam = 2
        return scale * init_weights(key_c, lam=lam, shape=(n, m))
    if init_weights == random.beta:
        a = random.choice(key_a, jnp.array(list(range(1, 10))))  # 1, 10 / 10.0
        b = random.choice(key_b, jnp.array(list(range(1, 10))))  # 1, 10 / 10.0
        # a = 2
        # b = 5
        return scale * init_weights(key_c, a=a, b=b, shape=(n, m))
    if init_weights == random.weibull_min:
        a = random.choice(key_a, jnp.array(list(range(2, 10))))  # 2, 10
        b = random.choice(key_b, jnp.array(list(range(2, 10))))  # 2, 10
        # a = 3
        # b = 2
        return scale * init_weights(key_c,
                                    scale=a, concentration=b, shape=(n, m))
    if random.gamma == init_weights:
        # a = 0.5
        a = random.choice(key_a, jnp.array(list(range(2, 10)))) / 10.0  # 2, 10
        return scale * random.gamma(key_c, a=a, shape=(n, m))
    if random.uniform == init_weights:
        return scale * init_weights(key_c, shape=(n, m))
    if random.normal == init_weights:
        return 1 * init_weights(key_c, shape=(n, m))
    return scale * init_weights(w_key, (n, m))


class mRNA_Model():
    def __init__(self, num_of_genes=10000,
                 last_function=jax.nn.softmax,
                 num_of_classes=2,
                 learning_rate=1e-2,
                 batch_size=16,
                 num_epochs=50,
                 metric_functions={},
                 loss_func=[],
                 decision_threshold=0.5,
                 train_set=None,
                 valid_set=None,
                 test_set=None,
                 save_model_per_epoch=True,
                 save_model_path="./weight",
                 history_path="./history_model.csv",
                 warmup_epochs=10) -> None:

        self.num_of_genes = num_of_genes
        self.key = random.PRNGKey(0)

        self.alterations_layers_params = [[random_layer(
            1, num_of_genes,
            self.key, scale=1,
            init_weights=init_weight_type), 0]
            for init_weight_type in # [random.normal]*3 +
                                    [random.gumbel]*3 +
                                    [random.weibull_min]*3 +
                                    [random.pareto]*3 +
                                    # [random.uniform]*3 +
                                    # [random.exponential]*1 +
                                    # [random.multivariate_normal]*3 +
                                    [random.beta]*3 +
                                    # [random.logistic]*3 +
                                    [random.gamma]*3 +
                                    [random.poisson]*3 +
                                    [random.maxwell]*3
        ]
        self.last_function = last_function
        self.num_of_classes = num_of_classes
        self.genes_to_consider = None
        self.history = defaultdict(list)
        self.train_set = train_set.copy()
        self.valid_set = valid_set.copy()
        self.test_set = test_set.copy()
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.metric_functions = metric_functions
        self.loss = loss_func
        self.ModelDevelopmentComplete = False
        self.decision_threshold = decision_threshold
        self.decision_threshold_history = []
        self.num_batches = len(self.train_set["X"]) // self.batch_size
        self.save_model_per_epoch = save_model_per_epoch
        self.save_model_path = save_model_path
        self.history_path = history_path
        self.DEBUG = False
        self.train_mode = True
        self.learning_rate = learning_rate
        self.warmpup_epochs = warmup_epochs
        self.history_gene_selection = []
        if self.warmpup_epochs > 0:
            lr = jax_opt.piecewise_constant([self.warmpup_epochs],
                                            [0.001, learning_rate])
        else:
            lr = self.learning_rate
        self.opt_init, self.opt_update, self.get_params = \
            jax_opt.rmsprop_momentum(lr, momentum=0.8)

    def Sinusoidal(self, x):
        half_dim = x.shape[-1] // 2
        rng_length = x.shape[-1]
        EMB = jnp.exp(jnp.arange(half_dim) *
                      -math.log(rng_length) / (half_dim - 1))
        EMB_S = jnp.sin(x[:, :half_dim] * EMB)
        x_h = x[:, half_dim:]
        half_dim = x_h.shape[-1]
        EMB = jnp.exp(jnp.arange(half_dim) *
                      -math.log(rng_length) / (half_dim - 1))
        EMB_C = jnp.cos(x_h * EMB)
        EMB = jnp.concatenate([EMB_S, EMB_C], axis=-1)
        return EMB

    def Manipulate_X(self, n, X):
        '''
        Get the Median Absolute Deviation

        Parameters
        ----------
        X : jnp.array
            Input array

        Returns
        -------
        float
            Median Absolute Deviation
        '''
        X = jnp.array(X)
        rn = rmd.randint(1, 1000)
        key = random.PRNGKey(rn)
        rn = rmd.randint(1, 1000)
        key_a = random.PRNGKey(rn)
        rn = rmd.randint(1, 1000)
        key_b = random.PRNGKey(rn)
        a = random.choice(key_a, jnp.array(list(range(1, 10))))  # / 10.0
        b = random.choice(key_b, jnp.array(list(range(1, 10))))  # / 10.0
        rsl = jnp.concatenate([random.beta(key, a, b, (2, n))*100,
                            random.beta(key, b, a, (2, n))*100,
                            random.beta(key, a * b, b ** a, (2, n))*100,
                            random.beta(key, a ** b, b * a, (2, n))*100])
        
        if (jnp.isnan(rsl).any()):
            rsl = jnp.nan_to_num(rsl, nan=0.0, posinf=None, neginf=None)
        q1 = jnp.quantile(rsl, 0.025)
        q2 = jnp.quantile(rsl, 0.975)
        rsl = jnp.clip(rsl, q1, q2)
        MAD = jnp.median(jnp.abs(rsl-jnp.median(rsl)))
        X_hist = jnp.histogram(X)
        get_top_3_indices = jnp.argpartition(X_hist[0], -3)[-3:]
        max_val = X_hist[1][get_top_3_indices][-1]
        min_val = X_hist[1][get_top_3_indices][0]
        X.at[X > max_val].set(X[X > max_val] + MAD)
        X.at[X < min_val].set(X[X < min_val] + MAD)
        X.at[X > min_val].set(X[X > min_val] + MAD/2)
        X.at[X < max_val].set(X[X < max_val] + MAD/2)
        return X

    def ModelDesignPhase(self, batches, step_i=1):
        '''
        Determmine the genes to consider
        The gene list determined by the correlation matrix
        We find the correlation matrix of the input data
        We assume that we have different noises in the training  with
        different distributions
        This step determines also the parameter capacity.
        This approach gives a dynamic feature of parameter capacity.
        '''
        # populate the input data per batch
        intersection_store = []
        for input_data in batches:
            if self.DEBUG:
                print(input_data.shape)
                print(jnp.max(input_data))
                print(jnp.min(input_data))
            ind_ix = input_data.argsort()  
            sorted_input_data = jnp.take_along_axis(input_data, ind_ix, axis=0)
            data_input = jnp.vstack([(w[0].T+sorted_input_data)
                                     for w in self.alterations_layers_params])
            data_input = data_input[:, ind_ix]
            data_input = jnp.round(data_input, 1)

            if self.DEBUG:
                print("data_input.shape", data_input.shape)
                print(data_input.shape)
                print(jnp.max(data_input))
                print(jnp.min(data_input))
                print(jnp.mean(data_input))
                print(jnp.median(data_input))
                print(jnp.std(data_input))
            # Get the correlation matrix
            g = jnp.abs(jnp.corrcoef(data_input.T))
            if self.DEBUG:
                print(g.shape)
                print(jnp.max(g))
                print(jnp.min(g))
                print(jnp.mean(g))
            # Remove diagnonal elements with 1 and lower triangle
            diag_elements = jnp.diag_indices_from(g)
            if self.DEBUG:
                print("diag_elements", diag_elements)
            out = g.at[diag_elements].set(0)
            if self.DEBUG:
                print("out", out.shape)
            out = jnp.triu(out)
            if self.DEBUG:
                print("out_2", out.shape)
            x_t = jnp.unique(jnp.argmax(out, axis=0))
            x_y = jnp.unique(jnp.argmax(out, axis=1))
            indices_genes = jnp.arange(g.shape[0])
            x_t = jnp.setdiff1d(indices_genes, x_t)
            x_y = jnp.setdiff1d(indices_genes, x_y)
            
            def get_indices(indices, axis=0):
                min_threshold = 0.967  # 9000  # 7500
                max_threshold = 0.980  #1.00 #0.9999#986  # 9900  # 9950
                select_index = []
                if axis == 0:
                    for x in indices:
                        max_ = jnp.max(out[x, :])
                        if max_ is None:
                            continue
                        if max_threshold > max_ > min_threshold:
                            select_index.append(x)
                if axis == 1:
                    for x in indices:
                        max_ = jnp.max(out[:, x])
                        if max_ is None:
                            continue
                        if max_threshold > max_ > min_threshold:
                            select_index.append(x)
                return select_index

            x_t = jnp.array(get_indices(x_t, axis=0))
            x_y = jnp.array(get_indices(x_y, axis=1))
            
            if self.DEBUG:
                print("x_t", x_t.shape)
                print("x_y", x_y.shape)
            #    1 2 3 4 5
            #   -----------
            # 1| 1 1 2 3 0    4
            # 2| 1 0 4 5 0    4
            # 3| 2 4 0 6 0    4
            # 4| 3 5 6 0 7    5
            # 5| 0 0 0 7 0    4
            # ----------------
            #    1 3 4 5 4
            # Intersection ==> 4,5
            intersect1d = jnp.unique(jnp.intersect1d(x_t, x_y)).flatten()
            intersection_store.extend(intersect1d)
        # GET THE INTERSECTION OF ALL THE BATCHES
        intersect1d = jnp.unique(jnp.array(intersection_store).flatten())
        if intersect1d.shape[0] == 0 and step_i > 1:
            self.genes_to_consider = self.history_gene_selection[-1]
        elif intersect1d.shape[0] == 0 and \
                len(self.history_gene_selection) == 0 and step_i == 1:
            self.genes_to_consider = jnp.array(list(range(self.num_of_genes)))
        else:
            self.genes_to_consider = intersect1d  # indices of selected genes
        self.history_gene_selection.append(self.genes_to_consider)
        self.num_of_genes_to_consider = self.history_gene_selection[-1].shape[0]
        # Initialize the model
        if step_i == 1:
            w1 = random_layer(1, self.num_of_genes, self.key, scale=1,
                              init_weights=random.normal)
            b1 = random_layer(1, 1, self.key, scale=1,
                              init_weights=random.normal)
            w2 = random_layer(self.num_of_genes,
                              self.num_of_classes,
                              self.key,
                              scale=1,
                              init_weights=random.normal)
            b2 = random_layer(1,
                              1,
                              self.key,
                              scale=1,
                              init_weights=random.normal)
            self.mlp_params = [[w1, b1], [w2, b2]]
            self.opt_state = self.opt_init(self.mlp_params)
        else:
            print("\n", self.genes_to_consider.shape,
                  self.mlp_params[0][0].shape,
                  self.mlp_params[1][0].shape)
        return 1

    def UpdateModel(self, x: jnp.array, y: jnp.array, step_i=0) -> Any:
        '''
        Run the gradient descent to update the model
        optimize the model weights using gradient descent

        Parameters
        ----------
        x : np.array
            The input data
        y : np.array
            The labels
        step_i : int, optional
            The step number. The default is 0.

        Returns
        -------
        (
            loss : float
                The loss value
            y_pred : np.array
                The prediction scores
            )
        '''
        def PredictAndLoss(params, x, y_true):
            # x = self.Sinusoidal(x)
            w1, b1 = params[0]
            w2, b2 = params[1]
            w1 = w1[self.genes_to_consider, :]  # .copy()#reduce
            w2 = w2[:, self.genes_to_consider]  # .copy()#reduce
            '''
            if self.train_mode:
                key_random = random.PRNGKey(rmd.randint(0, 100))
                # selected = rmd.uniform(1, 10)
                
                # if selected > 5:
                #     index = jax.random.choice(key_random, jnp.array([
                #         range(len(self.alterations_layers_params)-1)]))
                #     w = self.alterations_layers_params[int(index[0])].copy()
                #     w_B = w[0][self.genes_to_consider, :]
                #     x = (w_B.T*x)
                #     x = ((x - jnp.min(x, axis=1)[:, jnp.newaxis]) /
                #          (jnp.max(x, axis=1)[:, jnp.newaxis] -
                #           jnp.min(x, axis=1)[:, jnp.newaxis]))*100
                #     x = jnp.round(x, 1)
            '''
            lay1 = w1.T*x + b1
            lay1 = jax.nn.tanh(lay1)
            lay1 = jax.nn.softmax(lay1/(jnp.max(lay1)+1e-6), axis=0)
            lay1 = jax.nn.log_sigmoid(lay1*x + x)  # Residual
            lay2 = jnp.matmul(lay1, w2.T) + b2
            lst = self.last_function(lay2, axis=-1)
            y_pred = lst
            return self.loss[0](y_true, y_pred)

        y_true = y

        self.mlp_params = self.get_params(self.opt_state)

        loss, grads = value_and_grad(PredictAndLoss, argnums=0)(
            self.mlp_params,
            x,
            y_true)
        self.opt_state = self.opt_update(step_i, grads, self.opt_state)
        y_pred = jax.vmap(self.Predict)(x)
        b, _, c = y_pred.shape
        y_pred_b = y_pred.reshape((b, c))
        grads_a = jax.grad(lambda x, y: (sum((x[..., 1]) - y[..., 1])**2))(y_true, y_pred_b)
        grads_b = jax.grad(lambda x, y: (sum((x[..., 0]) - y[..., 0])**2))(y_true, y_pred_b)
        grads_c = jax.grad(lambda x, y: (sum((x[..., 1]) - y[..., 1])**2))(y_true, grads_a*(abs(y_pred_b-y_true)/(y_pred_b+y_true)))
        self.mlp_params = [(w - dw, b - db) for
                           (w, b), (dw, db) in zip(self.mlp_params, grads_a)]
        self.mlp_params = [(w - dw, b - db) for
                           (w, b), (dw, db) in zip(self.mlp_params, grads_b)]
        self.mlp_params = [(w - dw, b - db) for
                           (w, b), (dw, db) in zip(self.mlp_params, grads_c)]
        return loss, y_pred

    def RunStep(self, batch_x, batch_y, step_i=0, epoch=1):
        '''
        Run the model training for one step

        Parameters
        ----------
        batch_x : np.array
            The input data
        batch_y : np.array
            The labels
        step_i : int, optional
            The step number. The default is 0.
        epoch : int, optional
            The epoch number. The default is 1.

        Returns
        -------
        (
            loss: float
                The loss value
            y_pred : np.array
                The prediction scores
        )
        '''
        if not self.ModelDevelopmentComplete:
            steps_threshold = 50
            if epoch == 1 and step_i < steps_threshold:
                self.ModelDevelopmentComplete = False
                self.ModelDesignPhase(batch_x.copy(), step_i=step_i)
            if epoch == 1 and step_i >= steps_threshold:
                self.ModelDevelopmentComplete = True
                h_gene_indices_stacked = jnp.hstack(
                    self.history_gene_selection)
                print(h_gene_indices_stacked.shape)
                unique, counts = jnp.unique(h_gene_indices_stacked,
                                            return_counts=True)
                print(unique.shape)
                self.genes_to_consider = unique[
                    jnp.logical_and(counts >= int(round(steps_threshold*0.1,
                                                        0)),
                                    counts <= int(round(steps_threshold*1.00,
                                                        0)))]  # > 68

                print("Outcome:", self.genes_to_consider.shape)
                with open("history_gene_selection.pkl", "wb") as f:
                    pickle.dump(self.history_gene_selection, f)
                np.save("genes_to_consider.npy", self.genes_to_consider)

        batch_x = jnp.copy(batch_x[:, self.genes_to_consider])
        data_input = jnp.round(batch_x, 1)
        return self.UpdateModel(data_input, batch_y, step_i=step_i)

    def RunEpoch(self, epoch=1, progress_=None):
        '''
        Run the model training for one epoch
        '''
        loss_val = 0
        progress_ = tqdm(range(self.num_batches))
        counter = 1
        for batch in progress_:
            batch_images = self.train_set["X"][batch * self.batch_size:(batch+1) * self.batch_size]
            batch_labels = self.train_set["Y"][batch * self.batch_size:(batch+1) * self.batch_size]
            loss, y_pred = self.RunStep(batch_images,
                                        batch_labels,
                                        step_i=counter,
                                        epoch=epoch+1)

            loss_val += loss/batch_images.shape[0]
            metrics_txt = ""
            for key_metric in self.metric_functions:
                metric_vl = self.metric_functions[key_metric](batch_labels,
                                                              y_pred)
                metrics_txt += f" | {key_metric}: {metric_vl:.3f}"
            txt_description = f"Epoch {epoch+1}/{self.num_epochs} | Loss: \
            {loss_val / counter:.3f} {metrics_txt}"
            counter += 1
            progress_.set_description(txt_description)
            progress_.refresh()
        self.history["progress_description"].append(txt_description)
        self.history["loss"].append(loss_val)

    def Train(self):
        '''
        Execute the training process using the training set and validation set
        according to training hyperparameters
        '''
        self.train_mode = True
        for epoch in range(self.num_epochs):
            self.history["epoch"].append(epoch)
            self.RunEpoch(epoch=epoch)
            self.history["decision_threshold"].append(self.decision_threshold)
            self.history["num_of_genes_to_consider"].append(
                self.num_of_genes_to_consider)
            self.history["genes_to_consider"].append(self.genes_to_consider)
            self.Evaluate(self.valid_set["X"],
                          self.valid_set["Y"],
                          prefix="val_")
            if self.save_model_per_epoch:
                self.Save(self.save_model_path, epoch)
            pd.DataFrame(self.history).to_csv(self.history_path, index=False)

    def Test(self):
        '''
        Evaluate the model using the test set
        '''
        X = self.test_set["X"][:, self.genes_to_consider]
        print(jnp.max(X), jnp.min(X))
        y_pred = self.Predict(X)
        for key_metric in self.metric_functions:
            metric_vl = self.metric_functions[key_metric](self.test_set["Y"],
                                                          y_pred)
            print(f"test_{key_metric}: {metric_vl:.3f}")

    def Evaluate(self, data_input, y_true, prefix=""):
        '''
        FUNCTION TO EVALUATE THE MODEL
        '''
        self.train_mode = True
        y_pred = jax.vmap(self.Predict)(data_input[:, self.genes_to_consider])
        for key_metric in self.metric_functions:
            metric_vl = self.metric_functions[key_metric](y_true, y_pred)
            self.history[f"{prefix}{key_metric}"].append(metric_vl)
            print(f"{prefix}{key_metric}: {metric_vl:.3f}")

    def Predict(self, data_input):
        '''
        Predict the output of the model for a given input
        '''
        if self.train_mode:
            self.mlp_params = self.get_params(self.opt_state)
        if self.DEBUG:
            print(data_input.shape)
        w1, b1 = self.mlp_params[0]
        w2, b2 = self.mlp_params[1]
        w1 = w1[self.genes_to_consider, :]  # reduce
        w2 = w2[:, self.genes_to_consider]  # reduce
        lay1 = w1.T*data_input + b1
        lay1 = jax.nn.tanh(lay1)
        lay1 = jax.nn.softmax(lay1/(jnp.max(lay1) + 1e-6), axis=0)  # attention
        lay1 = jax.nn.log_sigmoid(lay1*data_input+data_input)  # Residual
        lay2 = jnp.matmul(lay1, w2.T) + b2
        lst = self.last_function(lay2, axis=-1)
        return lst

    def Save(self, path, marking):
        '''
        Save the model to a given path
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, f"model_{marking}.pkl"), "wb") as f:
            pickle.dump({"mlp_params": self.mlp_params,
                         "alterations_layers_params":
                             self.alterations_layers_params,
                         "genes": self.genes_to_consider}, f)

    def Load(self, path):
        '''
        load the model from a given path
        '''
        with open(path, "rb") as f:
            dict_ = pickle.load(f)
            self.mlp_params = dict_["mlp_params"]
            self.alterations_layers_params = dict_["alterations_layers_params"]
            self.genes_to_consider = dict_["genes"]