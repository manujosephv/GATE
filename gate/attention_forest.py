# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
import logging
from typing import Any, Callable, Dict
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

# from pytorch_tabular.utils import _initialize_layers
import random

# from pytorch_tabular.base_model import BaseModel
from pytorch_tabular.models import BaseModel
from .utils import entmax15, sparsemax, entmoid15, sparsemoid

logger = logging.getLogger(__name__)


# Redefining initialize layers from PyTorch Tabular to handle sequential objects
# Will be ported to PyTorch Tabular in new version
def _initialize_layers(activation, initialization, layers):
    if type(layers) == nn.Sequential:
        for layer in layers:
            if hasattr(layer, "weight"):
                _initialize_layers(activation, initialization, layer)
    else:
        if activation == "ReLU":
            nonlinearity = "relu"
        elif activation == "LeakyReLU":
            nonlinearity = "leaky_relu"
        else:
            if initialization == "kaiming":
                logger.warning(
                    "Kaiming initialization is only recommended for ReLU and LeakyReLU."
                )
                nonlinearity = "leaky_relu"
            else:
                nonlinearity = "relu"

        if initialization == "kaiming":
            nn.init.kaiming_normal_(layers.weight, nonlinearity=nonlinearity)
        elif initialization == "xavier":
            nn.init.xavier_normal_(
                layers.weight,
                gain=nn.init.calculate_gain(nonlinearity)
                if activation in ["ReLU", "LeakyReLU"]
                else 1,
            )
        elif initialization == "random":
            nn.init.normal_(layers.weight)


class NeuralDecisionStump(nn.Module):
    def __init__(
        self,
        n_features: int,
        binning_activation: Callable = entmax15,
        feature_mask_function: Callable = entmax15,
    ):
        super().__init__()
        self._num_cutpoints = 1
        self._num_leaf = 2
        self.n_features = n_features
        self.binning_activation = binning_activation
        self.feature_mask_function = feature_mask_function
        self._build_network()

    def _build_network(self):
        if self.feature_mask_function is not None:
            # sampling a random beta distribution
            # random distribution helps with diversity in trees and feature splits
            alpha = random.uniform(0.5, 10.0)
            beta = random.uniform(0.5, 10.0)
            # with torch.no_grad():
            feature_mask = (
                torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([beta]))
                .sample((self.n_features,))
                .squeeze(-1)
            )
            self.feature_mask = nn.Parameter(feature_mask, requires_grad=True)
        W = torch.linspace(
            1.0,
            self._num_cutpoints + 1.0,
            self._num_cutpoints + 1,
            requires_grad=False,
        ).reshape(1, 1, -1)
        self.register_buffer("W", W)

        cutpoints = torch.rand([self.n_features, self._num_cutpoints])
        # Append zeros to the beginning of each row
        cutpoints = torch.cat(
            [torch.zeros([self.n_features, 1], device=cutpoints.device), cutpoints], 1
        )
        self.cut_points = nn.Parameter(cutpoints, requires_grad=True)
        self.leaf_responses = nn.Parameter(
            torch.rand(self.n_features, self._num_leaf), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_mask_function is not None:
            feature_mask = self.feature_mask_function(self.feature_mask)
        # Repeat W for each batch size using broadcasting
        W = torch.ones(x.size(0), 1, 1, device=x.device) * self.W
        # Binning features
        x = torch.bmm(x.unsqueeze(-1), W) - self.cut_points.unsqueeze(0)
        x = self.binning_activation(x)  # , dim=-1)
        x = x * self.leaf_responses.unsqueeze(0)
        x = (x * feature_mask.reshape(1, -1, 1)).sum(dim=1)
        return x, feature_mask


class NeuralDecisionTree(nn.Module):
    def __init__(
        self,
        depth: int,
        n_features: int,
        dropout: float = 0,
        binning_activation: Callable = entmax15,
        feature_mask_function: Callable = entmax15,
    ):
        super().__init__()
        self.depth = depth
        self._num_cutpoints = 1
        self.n_features = n_features
        self._dropout = dropout
        self.binning_activation = binning_activation
        self.feature_mask_function = feature_mask_function
        self._build_network()

    def _build_network(self):
        for d in range(self.depth):
            for n in range(max(2 ** (d), 1)):
                self.add_module(
                    "decision_stump_{}_{}".format(d, n),
                    NeuralDecisionStump(
                        self.n_features + (2 ** (d) if d > 0 else 0),
                        self.binning_activation,
                        self.feature_mask_function,
                    ),
                )
        self.dropout = nn.Dropout(self._dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tree_input = x
        feature_masks = []
        for d in range(self.depth):
            layer_nodes = []
            layer_feature_masks = []
            for n in range(max(2 ** (d), 1)):
                leaf_nodes, feature_mask = self._modules[
                    "decision_stump_{}_{}".format(d, n)
                ](tree_input)
                layer_nodes.append(leaf_nodes)
                layer_feature_masks.append(feature_mask)
            layer_nodes = torch.cat(layer_nodes, dim=1)
            tree_input = torch.cat([x, layer_nodes], dim=1)
            feature_masks.append(layer_feature_masks)
        return self.dropout(layer_nodes), feature_masks


class GatedFeatureLearningUnit(nn.Module):
    def __init__(
        self,
        n_features_in: int,
        n_stages: int,
        feature_mask_function: Callable = entmax15,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features_in = n_features_in
        self.n_features_out = n_features_in
        self.feature_mask_function = feature_mask_function
        self._dropout = dropout
        self.n_stages = n_stages
        self._build_network()

    def _create_feature_mask(self):
        feature_masks = torch.cat(
            [
                torch.distributions.Beta(
                    torch.tensor([random.uniform(0.5, 10.0)]),
                    torch.tensor([random.uniform(0.5, 10.0)]),
                )
                .sample((self.n_features_in,))
                .squeeze(-1)
                for _ in range(self.n_stages)
            ]
        ).reshape(self.n_stages, self.n_features_in)
        return nn.Parameter(
            feature_masks,
            requires_grad=True,
        )

    def _build_network(self):
        self.W_in = nn.ModuleList(
            [
                nn.Linear(2 * self.n_features_in, 2 * self.n_features_in)
                for _ in range(self.n_stages)
            ]
        )
        self.W_out = nn.ModuleList(
            [
                nn.Linear(2 * self.n_features_in, self.n_features_in)
                for _ in range(self.n_stages)
            ]
        )

        self.feature_masks = self._create_feature_mask()
        self.dropout = nn.Dropout(self._dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for d in range(self.n_stages):
            feature = self.feature_mask_function(self.feature_masks[d]) * x
            h_in = self.W_in[d](torch.cat([feature, h], dim=-1))
            z = torch.sigmoid(h_in[:, : self.n_features_in])
            r = torch.sigmoid(h_in[:, self.n_features_in :])
            h_out = torch.tanh(self.W_out[d](torch.cat([r * h, x], dim=-1)))
            h = self.dropout((1 - z) * h + z * h_out)
        return h


class GatedAdditiveTrees(nn.Module):
    ACTIVATION_MAP = {
        "entmax": entmax15,
        "sparsemax": sparsemax,
        "softmax": nn.functional.softmax,
    }

    BINARY_ACTIVATION_MAP = {
        "entmoid": entmoid15,
        "sparsemoid": sparsemoid,
        "sigmoid": nn.functional.sigmoid,
    }

    def __init__(
        self,
        cat_embedding_dims: list,
        n_continuous_features: int,
        gflu_stages: int,
        num_trees: int,
        tree_depth: int,
        chain_trees: bool = True,
        gflu_dropout: float = 0.0,
        tree_dropout: float = 0.0,
        binning_activation: str = "entmoid",
        feature_mask_function: str = "softmax",
        batch_norm_continuous_input: bool = True,
    ):
        super().__init__()
        assert (
            binning_activation in self.BINARY_ACTIVATION_MAP.keys()
        ), f"`binning_activation should be one of {self.BINARY_ACTIVATION_MAP.keys()}"
        assert (
            feature_mask_function in self.ACTIVATION_MAP.keys()
        ), f"`feature_mask_function should be one of {self.ACTIVATION_MAP.keys()}"

        self.gflu_stages = gflu_stages
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.chain_trees = chain_trees
        self.gflu_dropout = gflu_dropout
        self.tree_dropout = tree_dropout
        self.binning_activation = self.BINARY_ACTIVATION_MAP[binning_activation]
        self.feature_mask_function = self.ACTIVATION_MAP[feature_mask_function]
        self.batch_norm_continuous_input = batch_norm_continuous_input
        self.n_continuous_features = n_continuous_features
        self.cat_embedding_dims = cat_embedding_dims
        self._embedded_cat_features = sum([y for x, y in cat_embedding_dims])
        self.n_features = self._embedded_cat_features + n_continuous_features
        self._build_network()

    def _build_network(self):
        # Embedding layers
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in self.cat_embedding_dims]
        )
        if self.batch_norm_continuous_input:
            self.normalizing_batch_norm = nn.BatchNorm1d(self.n_continuous_features)

        if self.gflu_stages > 0:
            self.gflus = GatedFeatureLearningUnit(
                n_features_in=self.n_features,
                n_stages=self.gflu_stages,
                feature_mask_function=self.feature_mask_function,
                dropout=self.gflu_dropout,
            )
        self.trees = nn.ModuleList(
            [
                NeuralDecisionTree(
                    depth=self.tree_depth,
                    n_features=self.n_features + 2**self.tree_depth * t
                    if self.chain_trees
                    else self.n_features,
                    dropout=self.tree_dropout,
                    binning_activation=self.binning_activation,
                    feature_mask_function=self.feature_mask_function,
                )
                for t in range(self.num_trees)
            ]
        )

    def unpack_input(self, x: Dict):
        continuous_data, categorical_data = x["continuous"], x["categorical"]
        if len(self.cat_embedding_dims) != 0:
            x = [
                embedding_layer(categorical_data[:, i])
                for i, embedding_layer in enumerate(self.embedding_layers)
            ]
            x = torch.cat(x, 1)

        if self.n_continuous_features != 0:
            if self.batch_norm_continuous_input:
                continuous_data = self.normalizing_batch_norm(continuous_data)

            if len(self.cat_embedding_dims) != 0:
                x = torch.cat([x, continuous_data], 1)
            else:
                x = continuous_data
        return x

    def forward(self, x: Dict):
        # Dict of (B, N1), (B,N2) --> (B, N1+N2)
        x = self.unpack_input(x)
        if self.gflu_stages > 0:
            x = self.gflus(x)
        # Decision Tree
        tree_outputs = []
        tree_feature_masks = []
        tree_input = x
        for i in range(self.num_trees):
            tree_output, feat_masks = self.trees[i](tree_input)
            tree_outputs.append(tree_output.unsqueeze(-1))
            tree_feature_masks.append(feat_masks)
            if self.chain_trees:
                tree_input = torch.cat([tree_input, tree_output], 1)
        return (torch.cat(tree_outputs, dim=-1), tree_feature_masks)


class GatedAdditiveTreeEnsembleBackbone(pl.LightningModule):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__()
        self.save_hyperparameters(config)
        self.kwargs = kwargs
        self._build_network()

    def _build_network(self):
        self.network = GatedAdditiveTrees(
            n_continuous_features=self.hparams.continuous_dim,
            cat_embedding_dims=self.hparams.embedding_dims,
            gflu_stages=self.hparams.gflu_stages,
            gflu_dropout=self.hparams.gflu_dropout,
            num_trees=self.hparams.num_trees,
            tree_depth=self.hparams.tree_depth,
            tree_dropout=self.hparams.tree_dropout,
            binning_activation=self.hparams.binning_activation,
            feature_mask_function=self.hparams.feature_mask_function,
            batch_norm_continuous_input=self.hparams.batch_norm_continuous_input,
            chain_trees=self.hparams.chain_trees,
        )
        self.output_dim = 2**self.hparams.tree_depth

    def forward(self, x: Dict):
        leaf_outputs, tree_feature_masks = self.network(x)
        return leaf_outputs, tree_feature_masks


class GatedAdditiveTreeEnsembleModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _build_network(self):
        self.backbone = GatedAdditiveTreeEnsembleBackbone(self.hparams)
        self.tree_attention = nn.MultiheadAttention(
            self.backbone.output_dim,
            1,
            dropout=self.hparams.tree_wise_attention_dropout,
        )
        head_input_dim = self.backbone.output_dim
        if self.hparams.share_head_weights:
            # Head
            self.head = nn.Sequential(
                nn.Dropout(self.hparams.dropout),
                nn.Linear(
                    head_input_dim,
                    self.hparams.output_dim,
                ),
            )
            _initialize_layers(
                self.hparams.activation, self.hparams.initialization, self.head
            )
        else:
            self.head = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Dropout(self.hparams.dropout),
                        nn.Linear(head_input_dim, self.hparams.output_dim),
                    )
                    for _ in range(self.hparams.num_trees)
                ]
            )
            [
                _initialize_layers(
                    self.hparams.activation, self.hparams.initialization, h
                )
                for h in self.head
            ]
        # random parameter with num_trees elements
        self.eta = nn.Parameter(torch.rand(self.hparams.num_trees, requires_grad=True))
        if self.hparams.task == "regression":
            self.T0 = nn.Parameter(
                torch.rand(self.hparams.output_dim), requires_grad=True
            )

    def apply_output_sigmoid_scaling(self, y_hat: torch.Tensor):
        if (self.hparams.task == "regression") and (
            self.hparams.target_range is not None
        ):
            for i in range(self.hparams.output_dim):
                y_min, y_max = self.hparams.target_range[i]
                y_hat[:, i] = y_min + nn.Sigmoid()(y_hat[:, i]) * (y_max - y_min)
        return y_hat

    def pack_output(
        self, y_hat: torch.Tensor, backbone_features: torch.tensor
    ) -> Dict[str, Any]:
        # if self.head is the Identity function it means that we cannot extract backbone features,
        # because the model cannot be divide in backbone and head (i.e. TabNet)
        if type(self.head) == nn.Identity:
            return {"logits": y_hat}
        else:
            return {"logits": y_hat, "backbone_features": backbone_features}

    def compute_backbone(self, x: Dict):
        # Returns output
        x, _ = self.backbone(x)
        return x

    def compute_head(
        self,
        backbone_features: torch.Tensor,
    ):
        # B x L x T
        if not self.hparams.share_head_weights:
            # B x T X Output
            y_hat = torch.cat(
                [
                    h(backbone_features[:, :, i]).unsqueeze(1)
                    for i, h in enumerate(self.head)
                ],
                dim=1,
            )
        else:
            # https://discuss.pytorch.org/t/how-to-pass-a-3d-tensor-to-linear-layer/908/6
            # B x T x L -> B x T x Output
            y_hat = self.head(backbone_features.transpose(2, 1))

        # applying weights to each tree and summing up
        # ETA
        y_hat = y_hat * self.eta.reshape(1, -1, 1)
        # summing up
        y_hat = y_hat.sum(dim=1)

        if self.hparams.task == "regression":
            y_hat = y_hat + self.T0
        # Sigmoid Scaling if a target range is provided (regression)
        y_hat = self.apply_output_sigmoid_scaling(y_hat)
        return self.pack_output(y_hat, backbone_features)

    def forward(self, x: Dict):
        x = self.compute_backbone(x)
        # if self.hparams.task == "ssl":
        #     return self.compute_ssl_head(x)
        return self.compute_head(x)

    def data_aware_initialization(self, datamodule):
        """Performs data-aware initialization for NDtree"""
        if self.hparams.task == "regression":
            logger.info("Data Aware Initialization....")
            # Need a big batch to initialize properly
            alt_loader = datamodule.train_dataloader(batch_size=2000)
            batch = next(iter(alt_loader))
            for k, v in batch.items():
                if isinstance(v, list) and (len(v) == 0):
                    # Skipping empty list
                    continue
                batch[k] = v.to(self.device)
            self.T0.data = torch.mean(batch["target"], dim=0)
