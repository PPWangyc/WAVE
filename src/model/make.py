from batcher import make_batcher
from decoder import make_decoder
from embedder import make_embedder
from embedder.unembedder import make_unembedder
from model.model import Model, DecoderModel, EmbedderModel, CustomModel, EncoderModel
from typing import Dict

def make_model(model_config: Dict=None):
    """Make model from model_config 
    (as generated by get_config()).
    """
    embedder = make_embedder(
        training_style=model_config["training_style"],
        architecture=model_config["architecture"],
        in_dim=model_config["parcellation_dim"],
        embed_dim=model_config["embedding_dim"],
        num_hidden_layers=model_config["num_hidden_layers_embedding_model"],
        dropout=model_config["dropout"],
        t_r_precision=model_config["tr_precision"],
        max_t_r=model_config["tr_max"],
        masking_rate=model_config["masking_rate"],
        n_positions=model_config["n_positions"]
    )
    decoder = make_decoder(
        architecture=model_config["architecture"],
        num_hidden_layers=model_config["num_hidden_layers"],
        embed_dim=model_config["embedding_dim"],
        num_attention_heads=model_config["num_attention_heads"],
        n_positions=model_config["n_positions"],
        intermediate_dim_factor=model_config["intermediate_dim_factor"],
        hidden_activation=model_config["hidden_activation"],
        dropout=model_config["dropout"],
        autoen_teacher_forcing_ratio=model_config["autoen_teacher_forcing_ratio"],
    )

    if model_config["embedding_dim"] != model_config["parcellation_dim"]:
        unembedder = make_unembedder(
            embed_dim=model_config["embedding_dim"],
            num_hidden_layers=model_config["num_hidden_layers_unembedding_model"],
            out_dim=model_config["parcellation_dim"],
            dropout=model_config["dropout"],
        )

    else:
        unembedder = None

    model = Model(
        embedder=embedder,
        decoder=decoder,
        unembedder=unembedder
    )

    return model

def make_decoder_model(model):
    return DecoderModel(model)

def make_embedder_model(model):
    return EmbedderModel(model)

def make_custom_model(model):
    return CustomModel(model)

def make_encoder_model(model):
    return EncoderModel(model)