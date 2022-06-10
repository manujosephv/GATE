from distutils.util import strtobool

import numpy as np
import pandas as pd
import plac
# import pytorch_lightning as pl
import torch
from pytorch_tabular import TabularModel
from pytorch_tabular.config import (
    DataConfig,
    ExperimentConfig,
    OptimizerConfig,
    TrainerConfig,
)
from sklearn.metrics import accuracy_score, f1_score

from gate.config import GatedAdditiveTreeEnsembleConfig
from gate.attention_forest import GatedAdditiveTreeEnsembleModel
from config.static_config import DATASET_MAP, LEARNING_RATE_SCHEDULER_MAP, OPTIMIZER_MAP


def load_data(data):
    if "target" in data["feature_names"]:
        data["feature_names"].remove("target")
    train_df = pd.DataFrame(data["X_train"], columns=data["feature_names"])
    train_df["target"] = data["y_train"]
    valid_df = pd.DataFrame(data["X_valid"], columns=data["feature_names"])
    valid_df["target"] = data["y_valid"]
    test_df = pd.DataFrame(data["X_test"], columns=data["feature_names"])
    test_df["target"] = data["y_test"]
    return (
        train_df,
        valid_df,
        test_df,
        data["feature_names"],
        data.get("num_classes", None),
    )


def calculate_metrics(y_true, y_pred, tag, task, average=None):
    if average is None:
        average = "macro" if len(np.unique(y_true)) > 2 else "binary"
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if y_true.ndim > 1:
        y_true = y_true.ravel()
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    if task == "classification":
        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average=average)
        print(f"{tag} Acc: {val_acc} | {tag} F1: {val_f1}")
        return val_acc, val_f1
    else:
        val_mse = np.mean(np.square(y_true - y_pred))
        val_mae = np.mean(np.abs(y_true - y_pred))
        print(f"{tag} MSE: {val_mse} | {tag} MAE: {val_mae}")
        return val_mse, val_mae


def get_data(dataset):
    data_dict = DATASET_MAP[dataset]
    data = data_dict["callable"](data_dict["path"])
    train_df, valid_df, test_df, feature_names, num_classes = load_data(data)
    del data
    return (
        data_dict,
        train_df,
        valid_df,
        test_df,
        feature_names,
        num_classes,
    )


def get_configs(
    feature_names,
    continuous_feature_transform,
    normalize_continuous_features,
    auto_lr_find,
    batch_size,
    max_epochs,
    use_gpu,
    early_stopping,
    optimizer,
    weight_decay,
    learning_rate_scheduler,
    data_dict,
    gflu_stages,
    gflu_dropout,
    num_trees,
    tree_depth,
    feature_mask_function,
    binning_activation,
    tree_dropout,
    share_head_weights,
    chain_trees,
    batch_norm_continuous_input,
    dropout,
    learning_rate,
    num_classes,
    tree_wise_attention,
    tree_wise_attention_dropout,
    dataset,
    checkpoints_path="saved_checkpoints/",
    checkpoints="valid_loss",  # valid_loss
    load_best=True,
    log_target="wandb",
    experiment_name=None,
    fast_dev_run=False,
    track_experiment=False
):
    data_config = DataConfig(
        target=["target"],
        continuous_cols=feature_names,
        categorical_cols=[],
        continuous_feature_transform=None
        if continuous_feature_transform == "none"
        else continuous_feature_transform,
        normalize_continuous_features=normalize_continuous_features,
        # num_workers=8,
    )
    trainer_config = TrainerConfig(
        auto_lr_find=auto_lr_find,
        batch_size=batch_size,
        max_epochs=max_epochs,
        auto_select_gpus=use_gpu,
        gpus=-1 if use_gpu else None,
        overfit_batches=0,
        deterministic=False,
        early_stopping_patience=3,
        early_stopping=None
        if early_stopping is None
        else early_stopping,  # "valid_accuracy",# "valid_loss",
        checkpoints=checkpoints,
        checkpoints_path=checkpoints_path,
        # best_model_path=best_model_path,
        load_best=load_best,
        # profiler="pytorch",
        fast_dev_run=fast_dev_run
    )
    opt = OPTIMIZER_MAP[optimizer]
    wd = weight_decay if weight_decay is not None else 0.0
    custom_opt = None
    if opt["custom"]:
        custom_opt = opt["param"]
    opt_params = {
        "weight_decay": wd,
    }
    lr_scheduler_dict = LEARNING_RATE_SCHEDULER_MAP[learning_rate_scheduler]
    optimizer_config = OptimizerConfig(
        optimizer=opt["param"] if not opt["custom"] else "adam",
        optimizer_params=opt_params,
        lr_scheduler=lr_scheduler_dict["param"],
        lr_scheduler_params=lr_scheduler_dict["args"],
    )
    _model_config = GatedAdditiveTreeEnsembleConfig

    model_params = dict(
        task=data_dict["task"],
        gflu_stages=gflu_stages,
        gflu_dropout=gflu_dropout,
        num_trees=num_trees,
        tree_depth=tree_depth,
        feature_mask_function=feature_mask_function,
        binning_activation=binning_activation,
        tree_dropout=tree_dropout,
        share_head_weights=share_head_weights,
        chain_trees=chain_trees,
        tree_wise_attention=tree_wise_attention,
        tree_wise_attention_dropout=tree_wise_attention_dropout,
        batch_norm_continuous_input=batch_norm_continuous_input,
        dropout=dropout,
        learning_rate=learning_rate,
        metrics=["f1", "accuracy"]
        if data_dict["task"] == "classification"
        else ["mean_squared_error", "mean_absolute_error"],
        metrics_params=[{"num_classes": num_classes, "average": "macro"}, {}]
        if data_dict["task"] == "classification"
        else [{}, {}],
    )

    model_config = _model_config(**model_params)
    run_name = f"GATE_{dataset}_ntrees_{num_trees}_depth_{tree_depth}_ft_{gflu_stages}"
    if track_experiment:
        experiment_config = ExperimentConfig(
            project_name=f"GATE_experiments_{dataset}",
            run_name=run_name if experiment_name is None else experiment_name,
            # exp_watch="gradients" if log_target == "wandb" else None,
            log_target=log_target,
            log_logits=False,
        )
    else:
        experiment_config = None
    return (
        data_config,
        trainer_config,
        optimizer_config,
        model_config,
        experiment_config,
        custom_opt,
        opt_params,
    )


@plac.pos("dataset", type=str, help="Dataset to use", choices=DATASET_MAP.keys())
@plac.opt(
    "continuous_feature_transform",
    type=str,
    abbrev="cft",
    help="Continuous feature transform to use",
    choices=["none", "yeo-johnson", "box-cox", "quantile_normal", "quantile_uniform"],
)
@plac.opt("use_gpu", help="Use GPU", type=strtobool, abbrev="gpu")
@plac.opt(
    "normalize_continuous_features",
    help="Normalize continuous features",
    type=strtobool,
    abbrev="ncf",
)
@plac.opt("learning_rate", type=float, help="Learning rate", abbrev="lr")
@plac.opt("batch_size", type=int, help="Batch size", abbrev="bs")
@plac.opt("auto_lr_find", help="Auto lr find", type=strtobool, abbrev="alr")
@plac.opt("max_epochs", type=int, help="Max epochs", abbrev="e")
@plac.opt("early_stopping", help="Early stopping. The metric to monitor for early stopping", type=str, abbrev="es")
@plac.opt("optimizer", type=str, help="Optimizer", choices=OPTIMIZER_MAP.keys())
@plac.opt("weight_decay", type=float, help="Weight decay", abbrev="wd")
@plac.opt(
    "learning_rate_scheduler",
    type=str,
    help="LR scheduler",
    choices=["none", "cosine"],
    abbrev="lrs",
)
@plac.opt(
    "gflu_stages",
    type=int,
    help="Feature abstraction stages",
    abbrev="fau",
)
@plac.opt(
    "gflu_dropout",
    type=float,
    help="Feature abstraction dropout",
    abbrev="fadp",
)
@plac.opt("num_trees", type=int, help="Num trees", abbrev="nt")
@plac.opt("tree_depth", type=int, help="Depth of Trees", abbrev="td")
@plac.opt(
    "tree_wise_attention", help="Tree wise attention", type=strtobool, abbrev="twa"
)
@plac.opt(
    "tree_wise_attention_dropout",
    help="Dropout for Tree wise attention",
    type=float,
    abbrev="twd",
)
@plac.opt(
    "feature_mask_function",
    type=str,
    help="Feature mask function",
    choices=["softmax", "entmax", "sparsemax"],
    abbrev="fmf",
)
@plac.opt(
    "binning_activation",
    type=str,
    help="Binning activation",
    choices=["entmoid", "sigmoid", "sparsemoid"],
    abbrev="ba",
)
@plac.opt("tree_dropout", type=float, help="Tree dropout", abbrev="tdr")
@plac.opt("share_head_weights", type=strtobool, help="Share head weights", abbrev="shw")
@plac.opt("chain_trees", type=strtobool, help="Chain trees", abbrev="ctree")
@plac.opt(
    "batch_norm_continuous_input",
    help="Batch norm continuous",
    type=strtobool,
    abbrev="bnci",
)
@plac.opt("dropout", type=float, help="Dropout", abbrev="drop")
@plac.opt("track_experiment", help="Track experiment", type=strtobool, abbrev="te")
@plac.opt("seed", help="Random Seed", type=int, abbrev="seed")
def run(
    dataset="FOREST",
    continuous_feature_transform="none",
    normalize_continuous_features=True,
    learning_rate=1e-3,
    auto_lr_find=False,
    batch_size=1024,
    max_epochs=100,
    early_stopping: str = "valid_loss",
    use_gpu: bool = True,
    optimizer: str = "adam",
    weight_decay: float = 1e-5,
    learning_rate_scheduler: str = "cosine",
    gflu_stages: int = 6,
    gflu_dropout: float = 0.0,
    num_trees: int = 20,
    tree_depth: int = 6,
    tree_wise_attention: bool = True,
    tree_wise_attention_dropout: float = 0.0,
    feature_mask_function: str = "entmax",
    binning_activation: str = "entmoid",
    tree_dropout: int = 0.0,
    share_head_weights=False,
    chain_trees=True,
    batch_norm_continuous_input: bool = False,
    dropout: float = 0.0,
    track_experiment: bool = False,
    checkpoints_path="saved_checkpoints/",
    experiment_name: str = None,
    fast_dev_run=False
):
    torch.cuda.empty_cache()
    data_dict, train_df, valid_df, test_df, feature_names, num_classes = get_data(
        dataset
    )
    (
        data_config,
        trainer_config,
        optimizer_config,
        model_config,
        experiment_config,
        custom_opt,
        opt_params,
    ) = get_configs(
        feature_names,
        continuous_feature_transform,
        normalize_continuous_features,
        auto_lr_find,
        batch_size,
        max_epochs,
        use_gpu,
        early_stopping,
        optimizer,
        weight_decay,
        learning_rate_scheduler,
        data_dict,
        gflu_stages,
        gflu_dropout,
        num_trees,
        tree_depth,
        feature_mask_function,
        binning_activation,
        tree_dropout,
        share_head_weights,
        chain_trees,
        batch_norm_continuous_input,
        dropout,
        learning_rate,
        num_classes,
        tree_wise_attention,
        tree_wise_attention_dropout,
        dataset,
        checkpoints_path,
        experiment_name=experiment_name,
        fast_dev_run=fast_dev_run,
        track_experiment=track_experiment
    )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        experiment_config=experiment_config if track_experiment else None,
        model_callable=GatedAdditiveTreeEnsembleModel,
    )

    tabular_model.fit(
        train=train_df,
        validation=valid_df,
        optimizer=custom_opt,
        optimizer_params=opt_params if custom_opt is not None else {},
        # callbacks=[RichProgressBar()],
    )
    if fast_dev_run:
        return "Model Fit Sucessfully"
    else:
        result = tabular_model.evaluate(test_df)
        print(result)
        return " | ".join([f"{k}:{v:.4f}" for k, v in result[0].items()])


if __name__ == "__main__":
    plac.call(run)
