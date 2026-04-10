import torch
import torch.nn as nn

from layers.Cross_Modal_Align import CrossModal
from layers.StandardNorm import Normalize


class Dual(nn.Module):
    def __init__(
        self,
        device="cuda:7",
        channel=32,
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        dropout_n=0.1,
        d_llm=768,
        e_layer=1,
        d_layer=1,
        d_ff=32,
        head=8,
        graph_adjacency=None,
        graph_steps=1,
        graph_alpha=0.5,
        task_name="legacy",
        input_features=1,
    ):
        super().__init__()

        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout_n = dropout_n
        self.d_llm = d_llm
        self.e_layer = e_layer
        self.d_layer = d_layer
        self.d_ff = d_ff
        self.head = head
        self.task_name = task_name
        self.input_features = input_features
        self.graph_steps = max(int(graph_steps), 0)
        self.use_revin = task_name != "multistock"

        if self.use_revin:
            self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
            self.series_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
            self.output_proj = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)
        else:
            flattened_dim = self.seq_len * self.input_features
            self.series_norm = nn.LayerNorm(flattened_dim).to(self.device)
            self.series_to_feature = nn.Linear(flattened_dim, self.channel).to(self.device)
            self.output_proj = nn.Linear(self.channel, 1, bias=True).to(self.device)

        self.ts_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel,
            nhead=self.head,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout_n,
        ).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(self.ts_encoder_layer, num_layers=self.e_layer).to(self.device)

        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm,
            nhead=self.head,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout_n,
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_encoder_layer, num_layers=self.e_layer).to(self.device)

        self.cross = CrossModal(
            d_model=self.num_nodes,
            n_heads=1,
            d_ff=self.d_ff,
            norm="LayerNorm",
            attn_dropout=self.dropout_n,
            dropout=self.dropout_n,
            pre_norm=True,
            activation="gelu",
            res_attention=True,
            n_layers=1,
            store_attn=False,
        ).to(self.device)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel,
            nhead=self.head,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout_n,
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.d_layer).to(self.device)

        if graph_adjacency is not None and self.graph_steps > 0:
            graph_adjacency = graph_adjacency.float().to(self.device)
            if graph_adjacency.shape != (self.num_nodes, self.num_nodes):
                raise ValueError(
                    f"Graph adjacency shape {tuple(graph_adjacency.shape)} does not match num_nodes={self.num_nodes}"
                )
            self.register_buffer("graph_adjacency", graph_adjacency)
            graph_alpha = float(min(max(graph_alpha, 1e-4), 1 - 1e-4))
            self.graph_gate = nn.Parameter(
                torch.logit(torch.tensor(graph_alpha, dtype=torch.float32, device=self.device))
            )
            self.use_graph = True
        else:
            self.register_buffer("graph_adjacency", None)
            self.graph_gate = None
            self.use_graph = False

    def param_num(self):
        return sum(param.nelement() for param in self.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _apply_graph(self, x):
        if not self.use_graph:
            return x

        propagated = x
        for _ in range(self.graph_steps):
            propagated = torch.matmul(propagated, self.graph_adjacency)

        mix_weight = torch.sigmoid(self.graph_gate)
        return (1 - mix_weight) * x + mix_weight * propagated

    def _encode_time_series(self, input_data):
        if self.task_name == "multistock":
            if input_data.dim() != 4:
                raise ValueError(f"Multi-stock mode expects [B, L, N, F], got {tuple(input_data.shape)}")
            batch_size, seq_len, num_nodes, num_features = input_data.shape
            if num_nodes != self.num_nodes:
                raise ValueError(f"Expected num_nodes={self.num_nodes}, got {num_nodes}")
            if num_features != self.input_features:
                raise ValueError(f"Expected input_features={self.input_features}, got {num_features}")
            input_data = input_data.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, seq_len * num_features)
            input_data = self.series_norm(input_data)
            return self.series_to_feature(input_data)

        input_data = self.normalize_layers(input_data, "norm")
        input_data = input_data.permute(0, 2, 1)
        return self.series_to_feature(input_data)

    def _prepare_embeddings(self, embeddings):
        if embeddings.dim() != 3:
            raise ValueError(f"Expected embeddings with 3 dims, got shape {tuple(embeddings.shape)}")
        if embeddings.size(1) == self.d_llm and embeddings.size(2) == self.num_nodes:
            return embeddings.permute(0, 2, 1).contiguous()
        if embeddings.size(1) == self.num_nodes and embeddings.size(2) == self.d_llm:
            return embeddings.contiguous()
        raise ValueError(
            f"Unexpected embedding shape {tuple(embeddings.shape)} for num_nodes={self.num_nodes} and d_llm={self.d_llm}"
        )

    def forward(self, input_data, input_data_mark, embeddings):
        del input_data_mark

        input_data = input_data.float()
        embeddings = embeddings.float()

        enc_out = self._encode_time_series(input_data)
        enc_out = self.ts_encoder(enc_out).permute(0, 2, 1)

        embeddings = self._prepare_embeddings(embeddings)
        embeddings = self.prompt_encoder(embeddings).permute(0, 2, 1)

        enc_out = self._apply_graph(enc_out)
        embeddings = self._apply_graph(embeddings)

        cross_out = self.cross(enc_out, embeddings, embeddings)
        cross_out = self._apply_graph(cross_out).permute(0, 2, 1)

        dec_out = self.decoder(cross_out, cross_out)
        dec_out = self.output_proj(dec_out)

        if self.task_name == "multistock":
            return dec_out.squeeze(-1)

        dec_out = dec_out.permute(0, 2, 1)
        dec_out = self.normalize_layers(dec_out, "denorm")
        return dec_out
