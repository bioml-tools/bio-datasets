"""End-to-end example of training with bio datasets.

Reproduce training of ProteinMPNN / LigandMPNN on LigandMPNN PDB splits.

`LigandMPNN was trained on protein assemblies in the PDB (as of Dec 16, 2022) determined
by X-ray crystallography or cryoEM to better than 3.5 Å resolution and with a total length of
less than 6,000 residues. The train/test split was based on protein sequences clustered at a
30% sequence identity cutoff. We evaluated LigandMPNN sequence design performance on
a test set of 317 protein structures containing a small molecule, 74 with nucleic acids, and 83
with a transition metal (Figure 2A). For comparison, we retrained ProteinMPNN on the same
training dataset of PDB biounits as LigandMPNN, except none of the context atoms were
provided during training.`

`The median sequence recoveries (ten designed sequences
per protein) near small molecules were 50.4% for Rosetta using the genpot energy function
(18), 50.4% for ProteinMPNN, and 63.3% for LigandMPNN. For residues near nucleotides,
median sequence recoveries were 35.2% for Rosetta (11) (using Rosetta energy optimized
for protein-DNA interfaces), 34.0% for ProteinMPNN, and 50.5% for LigandMPNN, and for
residues near metals, 36.0% for Rosetta (18), 40.6% for ProteinMPNN, and 77.5% for
LigandMPNN (Figure 2A). Sequence recoveries were consistently higher for LigandMPNN
over most proteins in the validation data set`

Splits available at: https://github.com/dauparas/LigandMPNN/tree/main/training

N.B. if we use an order-agnostic autoregressive model, then chain order doesn't matter.

reusing data across splits:
https://huggingface.co/docs/datasets/en/repository_structure
"""
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from bio_datasets.structure.biomolecule import BiomoleculeComplex


@dataclass
class ProteinMPNNConfig:
    num_layers: int = 3
    embed_dim: int = 128
    mlp_num_layers: int = 3
    dropout: float = 0.1
    aggregation_scale: float = 30.0
    num_atoms: int = 4
    num_rbf: int = 16
    residue_separation_embed_dim: int = 16
    max_residue_separation: int = 32
    gaussian_noise_std: float = 0.02
    num_neighbours: int = 30
    reference_distance_type: str = "ca"


def proteinmpnn_transforms(complex: BiomoleculeComplex, cfg: ProteinMPNNConfig):
    # TODO: apply transforms.
    return complex


def to_sparse_graph_features(complex: BiomoleculeComplex, cfg: ProteinMPNNConfig):
    # TODO: convert to sparse graph.
    return node_features, edge_features, edge_index


def pack_edges(
    node_h: torch.Tensor, edge_h: torch.Tensor, edge_idx: torch.LongTensor
) -> torch.Tensor:
    """Pack nodes and edge features into edge features. Source: Chroma
    https://github.com/generatebio/chroma/blob/main/chroma/layers/graph.py

    Expands each edge_ij by packing node i, node j, and edge ij into
    {node,node,edge}_ij.

    Args:
        node_h (torch.Tensor): Node features with shape
            `(num_batch, num_nodes, num_features_nodes)`.
        edge_h (torch.Tensor): Edge features with shape
            `(num_batch, num_nodes, num_neighbors, num_features_edges)`.
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_nodes, num_neighbors)`.

    Returns:
        edge_packed (torch.Tensor): Concatenated node and edge features with shape
            (num_batch, num_nodes, num_neighbors, num_features_nodes
                + 2*num_features_edges)`.
    """
    num_neighbors = edge_h.shape[2]
    node_i = node_h.unsqueeze(2).expand(-1, -1, num_neighbors, -1)
    node_j = collect_neighbors(node_h, edge_idx)
    edge_packed = torch.cat([node_i, node_j, edge_h], -1)
    return edge_packed


def collect_neighbors(node_h: torch.Tensor, edge_idx: torch.Tensor) -> torch.Tensor:
    """Collect neighbor node features as edge features. Source: Chroma
    https://github.com/generatebio/chroma/blob/main/chroma/layers/graph.py

    For each node i, collect the embeddings of neighbors {j in N(i)} as edge
    features neighbor_ij.

    Args:
        node_h (torch.Tensor): Node features with shape
            `(num_batch, num_nodes, num_features)`.
        edge_idx (torch.LongTensor): Edge indices for neighbors with shape
            `(num_batch, num_nodes, num_neighbors)`.

    Returns:
        neighbor_h (torch.Tensor): Edge features containing neighbor node information
            with shape `(num_batch, num_nodes, num_neighbors, num_features)`.

    Equivalent to protein mpnn gather_nodes
    """
    num_batch, num_nodes, num_neighbors = edge_idx.shape
    num_features = node_h.shape[2]

    # Flatten for the gather operation then reform the full tensor
    idx_flat = edge_idx.reshape([num_batch, num_nodes * num_neighbors, 1])
    idx_flat = idx_flat.expand(-1, -1, num_features)
    neighbor_h = torch.gather(node_h, 1, idx_flat)
    neighbor_h = neighbor_h.reshape((num_batch, num_nodes, num_neighbors, num_features))
    return neighbor_h


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int,
        activation: str = "gelu",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(in_features, hidden_features),
                nn.GELU(),
            ]
            + [
                nn.Linear(hidden_features, hidden_features),
                nn.GELU(),
            ]
            * (num_layers - 2)
            + [nn.Linear(hidden_features, out_features)]
        )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class ProteinMPNNNodeUpdate(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        mlp_hidden_dim: int,
        mlp_num_layers: int,
        dropout: float = 0.1,
        aggregation_scale: float = 30.0,
    ):
        super().__init__()
        self.aggregation_scale = aggregation_scale
        self.message_mlp = MLP(
            in_features=input_dim,
            hidden_features=mlp_hidden_dim,
            out_features=embed_dim,
            num_layers=mlp_num_layers,
        )
        self.update_linear = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        edge_index: torch.LongTensor,
        update_indices: Optional[torch.Tensor] = None,
        message_mask: Optional[torch.Tensor] = None,
    ):
        if update_indices is None:
            update_indices = torch.arange(node_h.shape[1])
        else:
            assert update_indices.ndim == 1

        edge_packed = pack_edges(node_h, edge_h, edge_index)[:, update_indices]
        messages = self.message_mlp(edge_packed).sum(-2) / self.aggregation_scale
        if message_mask is not None:
            messages = messages * message_mask[:, :, None].float()
        updated_nodes = self.layer_norm(
            node_h[:, update_indices] + self.dropout(messages)
        )
        node_updates = self.update_linear(updated_nodes)
        node_h[:, update_indices] = self.layer_norm(
            node_h[:, update_indices] + self.dropout(node_updates)
        )
        return node_h


class ProteinMPNNEdgeUpdate(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        mlp_hidden_dim: int,
        mlp_num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.message_mlp = MLP(
            in_features=input_dim,
            hidden_features=mlp_hidden_dim,
            out_features=embed_dim,
            num_layers=mlp_num_layers,
        )
        self.update_linear = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        edge_index: torch.LongTensor,
    ):
        edge_packed = pack_edges(node_h, edge_h, edge_index)
        messages = self.message_mlp(edge_packed)
        updated_edges = self.layer_norm(edge_h + self.dropout(messages))
        return updated_edges


class ProteinMPNNEncoderLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        mlp_hidden_dim: int,
        mlp_num_layers: int,
        dropout: float = 0.1,
        aggregation_scale: float = 30.0,
    ):
        super().__init__()
        self.node_update = ProteinMPNNNodeUpdate(
            input_dim=input_dim,
            embed_dim=embed_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_num_layers=mlp_num_layers,
            dropout=dropout,
            aggregation_scale=aggregation_scale,
        )
        self.edge_update = ProteinMPNNEdgeUpdate(
            input_dim=input_dim,
            embed_dim=embed_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_num_layers=mlp_num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        edge_index: torch.LongTensor,
        node_mask: Optional[torch.Tensor] = None,
        message_mask: Optional[torch.Tensor] = None,
    ):
        assert node_h.ndim == 3  # bsz, num_nodes, embed_dim
        assert edge_h.ndim == 4  # bsz, num_nodes, num_nodes, embed_dim

        node_h = self.node_update(node_h, edge_h, edge_index, message_mask=message_mask)
        # we mask both messages and nodes: masking messages ensures that masked nodes
        # edges cannot contribute to other nodes updates, and masking nodes ensures
        # that masked nodes remain as zeros
        node_h = node_h * node_mask[:, :, None].float()
        edge_h = self.edge_update(node_h, edge_h, edge_index)
        return node_h, edge_h


class ProteinMPNNDecoderLayer(nn.Module):
    """Autoregressive masking on node update; no edge update."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        mlp_hidden_dim: int,
        mlp_num_layers: int,
        dropout: float = 0.1,
        aggregation_scale: float = 30.0,
    ):
        super().__init__()
        self.node_update = ProteinMPNNNodeUpdate(
            input_dim=input_dim,
            embed_dim=embed_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_num_layers=mlp_num_layers,
            dropout=dropout,
            aggregation_scale=aggregation_scale,
        )

    def forward(
        self,
        decoder_h: torch.Tensor,
        structure_edge_h: torch.Tensor,
        sequence_edge_h: torch.Tensor,
        edge_index: torch.LongTensor,
        causal_mask_2d: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        update_indices: Optional[torch.Tensor] = None,
    ):
        # what's happening in the official code:
        # we have h_ESV and h_EXV; latter has sequence features zeroed out.
        # mask_bw is 1s for ar edges (i.e. where sequence can be attended to; mask fw where it cannot.)
        # then, in the decoder, the autoregressive masking applies to both sequence features and hidden states
        # that are influenced by the sequence features.

        sequence_edge_h = pack_edges(decoder_h, edge_h, edge_index)

        # we only need to mask the nodes to be updated:
        # we don't need to mask the message-sending structure nodes, because they have been masked out already.
        edge_h = node_mask[:, :, None].float() * torch.where(
            causal_mask_2d, sequence_edge_h, structure_edge_h
        )
        decoder_h = self.node_update(
            decoder_h, edge_h, edge_index, update_indices=update_indices
        )
        return decoder_h


class ProteinMPNNEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        embed_dim: int = 128,
        mlp_num_layers: int = 3,
        dropout: float = 0.1,
        aggregation_scale: float = 30.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                ProteinMPNNEncoderLayer(
                    input_dim=embed_dim
                    * 2,  # input layer concatenates node and edge features
                    embed_dim=embed_dim,
                    mlp_hidden_dim=embed_dim,
                    mlp_num_layers=mlp_num_layers,
                    dropout=dropout,
                    aggregation_scale=aggregation_scale,
                )
                for i in range(num_layers)
            ]
        )

    def forward(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        edge_index: torch.LongTensor,
        node_mask: Optional[torch.Tensor] = None,
    ):
        if node_mask is not None:
            node_h = node_h * node_mask[:, :, None].float()
            message_mask = collect_neighbors(node_mask[:, :, None], edge_index).squeeze(
                -1
            )  # b, num_nodes, num_neighbours
            message_mask = node_mask.unsqueeze(-1) * message_mask
        else:
            message_mask = None

        for layer in self.layers:
            node_h, edge_h = torch.utils.checkpoint.checkpoint(
                layer,
                node_h,
                edge_h,
                edge_index,
                node_mask=node_mask,
                message_mask=message_mask,
            )
        return node_h, edge_h


class ProteinMPNNDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        embed_dim: int = 128,
        mlp_num_layers: int = 3,
        dropout: float = 0.1,
        aggregation_scale: float = 30.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                ProteinMPNNDecoderLayer(
                    input_dim=embed_dim
                    * 2,  # input layer concatenates node, edge and sequence features
                    embed_dim=embed_dim,
                    mlp_hidden_dim=embed_dim,
                    mlp_num_layers=mlp_num_layers,
                    dropout=dropout,
                    aggregation_scale=aggregation_scale,
                )
                for i in range(num_layers)
            ]
        )
        self.sequence_embedding = nn.Embedding(21, embed_dim)
        self.output_projection = nn.Linear(embed_dim, 21)

    def forward(
        self,
        structure_h: torch.Tensor,
        sequence_index: torch.Tensor,
        edge_h: torch.Tensor,
        edge_index: torch.LongTensor,
        node_mask: Optional[torch.Tensor] = None,
        update_indices: Optional[torch.Tensor] = None,
    ):
        # TODO: handle random masks
        L = structure_h.shape[1]
        causal_mask_2d = torch.tril(
            torch.ones(L, L, dtype=torch.float32, device=structure_h.device),
            diagonal=-1,
        )  # cant see self
        # if sequence_mask_ij is 1 then we allow j to send message to i
        sequence_h = self.sequence_embedding(sequence_index)
        causal_mask_2d = torch.gather(
            causal_mask_2d, -1, edge_index
        )  # num_nodes, num_neighbours TODO: check
        sequence_edge_h = pack_edges(sequence_h, edge_h, edge_index)
        # TODO: unify these two (make sure concatenation order is same as order that decoder layer will use for sequence_edge_h)
        structure_edge_h = pack_edges(torch.zeros_like(sequence_h), edge_h, edge_index)
        structure_edge_h = pack_edges(structure_h, edge_h, edge_index)
        decoder_h = structure_h

        for layer in self.layers:
            decoder_h, edge_h = torch.utils.checkpoint.checkpoint(
                layer,
                decoder_h,
                structure_edge_h,
                sequence_edge_h,
                edge_index,
                node_mask=node_mask,
                causal_mask_2d=causal_mask_2d,
            )
        return self.output_projection(decoder_h)


class ProteinMPNN(nn.Module):
    def __init__(
        self,
        cfg: ProteinMPNNConfig,
    ):
        super().__init__()
        self.residue_separation_embedding = nn.Embedding(
            cfg.max_residue_separation, cfg.residue_separation_embed_dim
        )
        num_edge_features = (
            cfg.num_atoms**2 * cfg.num_rbf
        ) + cfg.residue_separation_embed_dim
        self.edge_embedding = nn.Linear(num_edge_features, cfg.embed_dim)
        self.encoder = ProteinMPNNEncoder(
            cfg.num_layers,
            cfg.embed_dim,
            cfg.mlp_num_layers,
            cfg.dropout,
            aggregation_scale=cfg.aggregation_scale,
        )
        self.decoder = ProteinMPNNDecoder(
            cfg.num_layers,
            cfg.embed_dim,
            cfg.mlp_num_layers,
            cfg.dropout,
            aggregation_scale=cfg.aggregation_scale,
        )
        self.edge_layer_norm = nn.LayerNorm(cfg.embed_dim)

    def forward(
        self,
        node_features: Dict[str, torch.Tensor],
        edge_features: Dict[str, torch.Tensor],
        edge_index: torch.LongTensor,
        node_mask: Optional[torch.Tensor] = None,
    ):
        _, num_nodes, _ = edge_index.shape  # bsz, num_nodes, num_neighbours
        node_h = torch.zeros(num_nodes, self.embed_dim, device=edge_index.device)
        edge_h = self.residue_separation_embedding(
            edge_features["residue_separation"].long()
        )
        edge_h = torch.cat([edge_h, edge_features["distance_rbfs"]], dim=-1)
        edge_h = self.edge_embedding(edge_h)
        edge_h = self.edge_layer_norm(edge_h)
        aa_ids = node_features["aa_ids"]
        aa_onehot = F.one_hot(aa_ids, num_classes=21)
        structure_h, edge_h = self.encoder(
            node_h, edge_h, edge_index, node_mask=node_mask
        )
        return self.decoder(
            structure_h, aa_onehot, edge_h, edge_index, node_mask=node_mask
        )

    def sample(
        self,
        node_features: Dict[str, torch.Tensor],
        edge_features: Dict[str, torch.Tensor],
        edge_index: torch.LongTensor,
    ):
        assert not "aa_ids" in node_features, "Must not provide aa ids for sampling"
        num_nodes, num_neighbours = edge_index.shape[0], edge_index.shape[1]
        node_h = torch.zeros(num_nodes, self.embed_dim, device=edge_index.device)
        edge_h = self.residue_separation_embedding(
            edge_features["residue_separation"].long()
        )
        raise NotImplementedError(
            "TODO: Implement efficient sampling via update_indices"
        )


class ProteinMPNNForInverseFolding(LightningModule):
    def __init__(self, cfg: ProteinMPNNConfig):
        super().__init__()
        self.model = ProteinMPNN(cfg.model)

    def forward(
        self,
        node_features: Dict[str, torch.Tensor],
        edge_features: Dict[str, torch.Tensor],
        edge_index: torch.LongTensor,  # could possibly be part of edge_features?
        node_mask: Optional[torch.Tensor] = None,
    ):
        return self.model(node_features, edge_features, edge_index, node_mask=node_mask)

    def training_step(self, batch, batch_idx):
        node_mask = batch["node_mask"]
        logits = self.forward(
            batch["node_features"], batch["edge_features"], batch["edge_index"]
        )
        targets = torch.where(node_mask.isnan().all(dim=-1), -100, batch["aa_ids"])
        # TODO: add scaling factor (why is this important?)
        loss = F.cross_entropy(logits, targets, ignore_index=-100)
        return loss