import pytest
import main


@pytest.mark.parametrize(
    "activation",
    [("softmax", "sigmoid"), ("entmax", "entmoid"), ("sparsemax", "sparsemoid")],
)
@pytest.mark.parametrize("dataset", ["A9A", "FOREST"])
def test_GATE(
    dataset,
    activation,
):
    results = main.run(
        dataset=dataset,
        continuous_feature_transform="none",
        normalize_continuous_features=True,
        learning_rate=1e-3,
        auto_lr_find=False,
        batch_size=1024,
        max_epochs=1,
        early_stopping="valid_loss",
        use_gpu=False,
        optimizer="adam",
        weight_decay=1e-5,
        learning_rate_scheduler="cosine",
        gflu_stages=1,
        gflu_dropout=0.0,
        num_trees=5,
        tree_depth=2,
        tree_wise_attention=True,
        tree_wise_attention_dropout=0.0,
        feature_mask_function=activation[0],
        binning_activation=activation[1],
        tree_dropout=0.0,
        share_head_weights=False,
        chain_trees=True,
        batch_norm_continuous_input=False,
        dropout=0.0,
        track_experiment=False,
        checkpoints_path="saved_checkpoints/",
        experiment_name=None,
        fast_dev_run=True,
    )
    assert results is not None
