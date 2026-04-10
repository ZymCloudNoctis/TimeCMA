import os

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer


class GenPromptEmb(nn.Module):
    def __init__(
        self,
        data_path="FRED",
        model_name="gpt2",
        device="cuda:0",
        input_len=96,
        d_model=768,
        layer=12,
        divide="train",
        freq="h",
        task_name="legacy",
        stock_codes=None,
        prompt_feature_index=0,
    ):
        super(GenPromptEmb, self).__init__()
        self.data_path = data_path
        self.data_name = os.path.basename(data_path).replace(".csv", "")
        self.device = device
        self.input_len = input_len
        self.model_name = model_name
        self.d_model = d_model
        self.layer = layer
        self.divide = divide
        self.freq = freq
        self.task_name = task_name
        self.stock_codes = stock_codes or []
        self.prompt_feature_index = prompt_feature_index
        self.last_index = self.input_len - 1

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name).to(self.device)

    def _format_datetime(self, in_data_mark, sample_index, time_index):
        year = int(in_data_mark[sample_index, time_index, 0])
        month = int(in_data_mark[sample_index, time_index, 1])
        day = int(in_data_mark[sample_index, time_index, 2])

        if self.freq == "d":
            return f"{year:04d}-{month:02d}-{day:02d}"

        hour = int(in_data_mark[sample_index, time_index, 4])
        minute = int(in_data_mark[sample_index, time_index, 5])
        if self.freq in {"h"}:
            return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:00"
        return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"

    def _legacy_prompt(self, in_data, in_data_mark, sample_index, node_index):
        values = in_data[sample_index, :, node_index].flatten().tolist()
        values_str = ", ".join([f"{value:.4f}" for value in values])
        trend_value = torch.sum(torch.diff(in_data[sample_index, :, node_index].flatten()))
        start_date = self._format_datetime(in_data_mark, sample_index, 0)
        end_date = self._format_datetime(in_data_mark, sample_index, self.last_index)
        frequency_text = {
            "d": "day",
            "h": "hour",
            "t": "15 minutes",
        }.get(self.freq, "time step")
        return (
            f"From {start_date} to {end_date}, the values were {values_str} every {frequency_text}. "
            f"The total trend value was {trend_value.item():.4f}"
        )

    def _multistock_prompt(self, in_data, in_data_mark, sample_index, node_index):
        values = in_data[sample_index, :, node_index, self.prompt_feature_index].flatten().tolist()
        values_str = ", ".join([f"{value:.4f}" for value in values])
        trend_value = torch.sum(torch.diff(in_data[sample_index, :, node_index, self.prompt_feature_index].flatten()))
        stock_code = self.stock_codes[node_index] if node_index < len(self.stock_codes) else f"stock_{node_index}"
        start_date = self._format_datetime(in_data_mark, sample_index, 0)
        end_date = self._format_datetime(in_data_mark, sample_index, self.last_index)
        return (
            f"For stock {stock_code}, from {start_date} to {end_date}, the closing prices were {values_str}. "
            f"The total price trend value was {trend_value.item():.4f}"
        )

    def _prepare_prompt(self, in_data, in_data_mark, sample_index, node_index):
        if in_data.dim() == 4 or self.task_name == "multistock":
            prompt = self._multistock_prompt(in_data, in_data_mark, sample_index, node_index)
        else:
            prompt = self._legacy_prompt(in_data, in_data_mark, sample_index, node_index)
        return self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

    def forward(self, tokenized_prompt):
        with torch.no_grad():
            prompt_embeddings = self.model(tokenized_prompt).last_hidden_state
        return prompt_embeddings

    def generate_embeddings(self, in_data, in_data_mark):
        if in_data.dim() == 4:
            batch_size, _, num_nodes, _ = in_data.shape
        elif in_data.dim() == 3:
            batch_size, _, num_nodes = in_data.shape
        else:
            raise ValueError(f"Unsupported input shape for prompt generation: {tuple(in_data.shape)}")

        tokenized_prompts = []
        max_token_count = 0
        for sample_index in range(batch_size):
            for node_index in range(num_nodes):
                tokenized_prompt = self._prepare_prompt(in_data, in_data_mark, sample_index, node_index)
                max_token_count = max(max_token_count, tokenized_prompt.shape[1])
                tokenized_prompts.append((sample_index, node_index, tokenized_prompt))

        in_prompt_emb = torch.zeros(
            (batch_size, num_nodes, max_token_count, self.d_model),
            dtype=torch.float32,
            device=self.device,
        )

        for sample_index, node_index, tokenized_prompt in tokenized_prompts:
            prompt_embeddings = self.forward(tokenized_prompt)
            padding_length = max_token_count - tokenized_prompt.shape[1]
            if padding_length > 0:
                last_token_embedding = prompt_embeddings[:, -1:, :].repeat(1, padding_length, 1)
                prompt_embeddings = torch.cat([prompt_embeddings, last_token_embedding], dim=1)
            in_prompt_emb[sample_index, node_index] = prompt_embeddings.squeeze(0)

        return in_prompt_emb[:, :, -1, :]
