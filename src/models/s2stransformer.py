import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()

    def forward(self, x):
        seq_len, batch_size, feature_dim = x.size()

        position = torch.arange(0, seq_len).unsqueeze(1).float().to(x.device)
        div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * -(math.log(10000.0) / feature_dim)).to(x.device)
        pe = torch.zeros(seq_len, 1, feature_dim).to(x.device)
        pe[:, 0, 0::2] = torch.sin(position * div_term[:feature_dim//2])
        pe[:, 0, 1::2] = torch.cos(position * div_term[:feature_dim//2])
        pe = pe.unsqueeze(0)

        # Add positional encoding
        x_with_pe = (x + pe).squeeze(0)

        return x_with_pe

class Seq2SeqTransformer(pl.LightningModule):
    def __init__(self, token_vocab_size, docid_vocab_size, d_model=256, nhead=4, num_layers=3, conv_channels=32, kernel_size=3, max_len=512):
        super(Seq2SeqTransformer, self).__init__()

        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.validation_accuracy_outputs = []

        # Embedding layer
        self.embedding = nn.Embedding(token_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer layers
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=False
        )

        # Convolutional layer
        self.conv1d = nn.Conv1d(in_channels=d_model, out_channels=conv_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)  # catch local pattern (L order-preserving tokens)

        # Linear layer
        self.fc = nn.Linear(conv_channels, docid_vocab_size)

    def forward(self, input_ids, target_ids):
        # Embedding
        input_embedding = self.embedding(input_ids)
        target_embedding = self.embedding(target_ids)

        # Add positional encoding
        input_embedding = self.positional_encoding(input_embedding)
        target_embedding = self.positional_encoding(target_embedding)


        input_mask = (input_ids != 0).unsqueeze(1).float()  # Mask for input sequence
        target_mask = (target_ids != 0).unsqueeze(1).float()  # Mask for target sequence
        # Pad the attention masks to have square dimensions
        input_mask = F.pad(input_mask, (0, target_ids.size(1)))
        target_mask = F.pad(target_mask, (0, input_ids.size(1)))
        # Concatenate the attention masks
        attn_mask = torch.cat([input_mask, target_mask], dim=2)
        attn_mask = attn_mask.expand(attn_mask.shape[0],attn_mask.shape[0],attn_mask.shape[2])
        attn_mask = attn_mask.permute(2,0,1)

        # Transformer with concatenated attention mask
        output_transformer = self.transformer(input_embedding, target_embedding, attn_mask)

        # Convolutional layer
        output_conv = self.conv1d(output_transformer.permute(1, 2, 0))
        output_conv = F.relu(output_conv)

        # Global max pooling
        output_pooled, _ = torch.max(output_conv, dim=2)

        # Linear layer
        output = self.fc(output_pooled)

        return output

    def training_step(self, batch, batch_idx):
        input_ids, target_ids, labels = batch
        target_ids = target_ids.permute(1, 0)
        input_ids = input_ids.permute(1, 0)
        output = self(input_ids, target_ids)

        # Assuming target_ids are integer labels
        loss = F.cross_entropy(output.view(-1, output.size(-1)), labels.view(-1))

        # Compute accuracy
        accuracy = (torch.argmax(output, dim=-1) == labels.view(-1)).float().mean()

        self.train_step_outputs.append(loss)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, target_ids, labels = batch
        target_ids = target_ids.permute(1, 0)
        input_ids = input_ids.permute(1, 0)
        output = self(input_ids, target_ids)

        loss = F.cross_entropy(output.view(-1, output.size(-1)), labels.view(-1))

        # Compute accuracy
        accuracy = (torch.argmax(output, dim=-1) == labels.view(-1)).float().mean()

        self.validation_step_outputs.append(loss)
        self.validation_accuracy_outputs.append(accuracy)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        if not len(self.train_step_outputs) == 0:
            epoch_average_train = torch.stack(self.train_step_outputs).mean()
            self.log("train_epoch_average", epoch_average_train)
            print("train_loss_avg: ", epoch_average_train)
            self.train_step_outputs.clear()
        if not len(self.validation_step_outputs) == 0:
            epoch_average = torch.stack(self.validation_step_outputs).mean()
            self.log("validation_epoch_average", epoch_average)
            print("val_loss_avg: ", epoch_average)
            self.validation_step_outputs.clear()

            self.validation_step_outputs.clear()
            accuracy_avg = sum(self.validation_accuracy_outputs) / len(self.validation_accuracy_outputs)
            print("accuracy: ", accuracy_avg)
            self.validation_accuracy_outputs.clear()


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=5e-2)
        return optimizer