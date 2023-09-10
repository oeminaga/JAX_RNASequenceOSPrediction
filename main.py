import os
from typing import Any
import jax
import pandas as pd
import argparse
import numpy as np
import models
import yaml
import jax.numpy as jnp
from sklearn.metrics import roc_auc_score

jax.config.update('jax_platform_name', 'cpu')

parser = argparse.ArgumentParser(description='Train mRNA model. Use cgf.yaml' +
                                 'to configure the pipeline.')
parser.add_argument('--full_pipeline', action='store_true', help='run a full' +
                    'pipeline including training, validation, and testing' +
                    ', please specify the data path in config.cfg')
parser.add_argument('--test', action='store_true', help='test a model')
print("Loading configuration file...")
# a function that load data and initialize model for training,
# validation and testing
with open("./cfg.yaml", 'r') as f:
    config = yaml.safe_load(f)

y_col = config['data']['y_col']
exclude_cols = config["data"]["exclude_cols"]
train_data_path = config["data"]["train_data"]
valid_data_path = config["data"]["valid_data"]
test_data_path = config["data"]["test_data"]
model_num_of_classes = config["model"]["num_of_classes"]
save_model_per_epoch = config["model"]["save_model_per_epoch"]
save_model_path = config["model"]["save_model_path"]
batch_size = config["model"]["batch_size"]
num_epochs = config["model"]["num_epochs"]
history_path = config["model"]["history_path"]
learning_rate = config["model"]["learning_rate"]
warmup_epochs = config["model"]["warmup_epochs"]


def one_hot(y: Any, num_of_classes: int) -> Any:
    '''
    Convert y to one-hot encoding
    param y: labels
    param num_of_classes: number of classes
    '''
    y_one_hot = np.zeros((y.shape[0], num_of_classes))
    y_one_hot[np.arange(y.shape[0]), y] = 1
    return y_one_hot


print("Loading data...")


def LoadData(data_path, y_col, exclude_cols, num_of_classes=2, repeat=1):
    '''
    Load data from data_path

    '''
    data = pd.read_csv(data_path)
    data = data.drop(exclude_cols, axis=1)
    data = data.sample(frac=1)  # shuffle
    data = data.reset_index(drop=True)
    y = data[y_col]
    X = data.drop(y_col, axis=1)
    # if repeat>1:
    #    X=X.sample(frac = repeat,replace=True) #repeat
    X.fillna(X.mean(numeric_only=True).round(4), inplace=True)
    X = X.rank(axis=1, pct=True)
    X = X.round(2)*100  # regularize
    X = X.to_numpy()
    print(data_path, np.max(X), np.min(X), np.mean(X), np.std(X), np.median(X))
    return {"X": X, "Y": one_hot(y.to_numpy(), num_of_classes=num_of_classes)}, X.shape[1]


if os.path.exists(train_data_path):
    train_data, gen_num = LoadData(train_data_path,
                                   y_col,
                                   exclude_cols,
                                   repeat=16)
    print(train_data["X"].shape)
if os.path.exists(valid_data_path):
    valid_data, gen_num = LoadData(valid_data_path, y_col, exclude_cols)
if os.path.exists(test_data_path):
    test_data, gen_num = LoadData(test_data_path, y_col, exclude_cols)


def mse(y_true, y_pred):
    '''
    Calculate accuracy
    '''
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return jnp.mean(jnp.square(y_true-y_pred))


def AUC(y_true, y_pred):
    '''
    Calculate AUC
    '''
    return roc_auc_score(y_true, y_pred)


def F1(y_true, y_pred):
    '''
    Calculate F1
    '''
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return 2*jnp.sum(y_true*y_pred)/(jnp.sum(y_true)+jnp.sum(y_pred))


def F1_50(y_true, y_pred):
    '''
    Calculate F1
    '''
    y_true = y_true[..., 1].flatten() > 0.5
    y_pred = y_pred[..., 1].flatten() > 0.5
    return 2*jnp.sum(y_true*y_pred)/(jnp.sum(y_true)+jnp.sum(y_pred))


def F1_25(y_true, y_pred):
    '''
    Calculate F1
    '''
    y_true = y_true[..., 1].flatten() > 0.25
    y_pred = y_pred[..., 1].flatten() > 0.25
    return 2*jnp.sum(y_true*y_pred)/(jnp.sum(y_true)+jnp.sum(y_pred))


def F1_75(y_true, y_pred):
    '''
    Calculate F1
    '''
    y_true = y_true[..., 1].flatten() > 0.75
    y_pred = y_pred[..., 1].flatten() > 0.75
    return 2*jnp.sum(y_true*y_pred)/(jnp.sum(y_true)+jnp.sum(y_pred))


def F1_25_0(y_true, y_pred):
    '''
    Calculate F1
    '''
    y_true = y_true[..., 0].flatten() > 0.25
    y_pred = y_pred[..., 0].flatten() > 0.25
    return 2*jnp.sum(y_true*y_pred)/(jnp.sum(y_true)+jnp.sum(y_pred))


def F1_50_0(y_true, y_pred):
    '''
    Calculate F1
    '''
    y_true = y_true[..., 0].flatten() > 0.5
    y_pred = y_pred[..., 0].flatten() > 0.5
    return 2*jnp.sum(y_true*y_pred)/(jnp.sum(y_true)+jnp.sum(y_pred))


def F1_75_0(y_true, y_pred):
    '''
    Calculate F1
    '''
    y_true = y_true[..., 0].flatten() > 0.75
    y_pred = y_pred[..., 0].flatten() > 0.75
    return 2*jnp.sum(y_true*y_pred)/(jnp.sum(y_true)+jnp.sum(y_pred))


def CategoricalCrossEntropy(y_true, y_pred):
    '''
    Calculate categorical cross entropy
    '''
    y_pred = jnp.clip(y_pred, 1e-7, 1.0 - 1e-7)
    loss = -jnp.sum(y_true*jnp.log(y_pred), axis=1)
    return jnp.mean(loss)


key = jax.random.PRNGKey(0)


def nll3_p(y_true, y_pred):
    """ Negative log likelihood. - Bernoulli """
    likelihood_a = jax.scipy.stats.bernoulli.cdf(key, y_pred)
    likelihood_b = jax.scipy.stats.bernoulli.cdf(key, y_true)
    loss_ = CategoricalCrossEntropy(likelihood_b, likelihood_a)
    return loss_


def nll3(y_true, y_pred):
    """ Negative log likelihood. Logistic SF"""
    likelihood_a = jax.scipy.stats.logistic.sf(y_pred, loc=0.5, scale=2)
    likelihood_b = jax.scipy.stats.logistic.sf(y_true, loc=0.5, scale=2)
    loss_ = CategoricalCrossEntropy(likelihood_b, likelihood_a)
    return loss_


def CCE_MSE(y_true, y_pred):
    '''
    Calculate categorical cross entropy + MSE
    '''
    return 0.70*CategoricalCrossEntropy(y_true, y_pred)+0.30*mse(y_true,
                                                                 y_pred)


if __name__ == "__main__":
    # Parse input arguments
    args = parser.parse_args()
    print(args)
    if args.full_pipeline:
        print("Loading model...")
        model = models.mRNA_Model(
            last_function=jax.nn.softmax,
            num_of_genes=gen_num,
            num_of_classes=model_num_of_classes,
            save_model_per_epoch=save_model_per_epoch,
            save_model_path=save_model_path,
            batch_size=batch_size,
            num_epochs=num_epochs,
            history_path=history_path,
            train_set=train_data,
            valid_set=valid_data,
            test_set=test_data,
            metric_functions={
                "mse": mse,
                "f1": F1,
                "f1_50": F1_50,
                "f1_25": F1_25,
                "f1_75": F1_75,
                "f1_25_0": F1_25_0,
                "f1_50_0": F1_50_0,
                "f1_75_0": F1_75_0
                },
            loss_func=[nll3],
            decision_threshold=0.8,
            learning_rate=learning_rate,
            warmup_epochs=warmup_epochs
        )
        print("Start training")
        model.Train()
        print("Start testing")
        0
        model.Test()
    elif args.test:
        path_ = os.path.join(save_model_path, "model.pkl")
    columns = pd.read_csv("./data/train_data.csv").columns
    col_select = np.load("./genes_to_consider.npy")
    print(columns[col_select])
