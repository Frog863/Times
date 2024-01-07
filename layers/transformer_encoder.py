import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
import math
class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout_rate):
        super(TransformerEncoder, self).__init__()

        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.finall_projection = nn.Linear(hidden_size, input_size)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(input_size,hidden_size, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])

    def forward(self, input_q,input_k,input_v):
        embedded_inputs_q = self.embedding(input_q)
        embedded_inputs_k = self.embedding(input_k)
        embedded_inputs_v = self.embedding(input_v)

        encoded_inputs_q = self.positional_encoding(embedded_inputs_q)
        encoded_inputs_k = self.positional_encoding(embedded_inputs_k)
        encoded_inputs_v = self.positional_encoding(embedded_inputs_v)

        output_q = self.dropout(encoded_inputs_q)
        output_k = self.dropout(encoded_inputs_k)
        output_v = self.dropout(encoded_inputs_v)



        for layer in self.encoder_layers:
            output = layer(output_q,output_k,output_v)
        output =self.finall_projection(output)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_size,hidden_size, num_heads, dropout_rate):
        super(TransformerEncoderLayer, self).__init__()

        self.multihead_attention = MultiheadAttention(input_size,hidden_size, num_heads, dropout_rate)
        self.feedforward = FeedForward(hidden_size, dropout_rate)

    def forward(self, input_q,input_k,input_v):
        attn_output = self.multihead_attention(input_q,input_k,input_v)
        residual_output = input_q + attn_output

        feedforward_output = self.feedforward(residual_output)
        output = residual_output + feedforward_output

        return output


class MultiheadAttention(nn.Module):
    def __init__(self, input_size,hidden_size, num_heads, dropout_rate):
        super(MultiheadAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_q,input_k,input_v):
        batch_size, seq_len, _ = input_k.size()

        query = self.query_projection(input_q).view(batch_size, seq_len, self.num_heads,
                                                   self.hidden_size // self.num_heads).transpose(1, 2)
        key = self.key_projection(input_k).view(batch_size, seq_len, self.num_heads,
                                               self.hidden_size // self.num_heads).transpose(1, 2)
        value = self.value_projection(input_v).view(batch_size, seq_len, self.num_heads,
                                                   self.hidden_size // self.num_heads).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden_size // self.num_heads))
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value).transpose(1, 2).contiguous().view(batch_size, seq_len,
                                                                                         self.hidden_size)
        output = self.output_projection(context)

        return output


class FeedForward(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.linear2 = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        output = self.linear1(inputs)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_seq_length=2048):
        super(PositionalEncoding, self).__init__()

        position_enc = torch.zeros(max_seq_length, hidden_size)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))

        position_enc[:, 0::2] = torch.sin(position * div_term)
        position_enc[:, 1::2] = torch.cos(position * div_term)
        self.dropout = torch.dropout(0.3)
        self.register_buffer('position_enc', position_enc.unsqueeze(0))

    def forward(self, inputs):
        seq_length = inputs.size(1)

        return self.dropout(inputs + self.position_enc[:, :seq_length])


