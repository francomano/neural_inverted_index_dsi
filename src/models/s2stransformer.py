import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl


class Seq2SeqTransformer(pl.LightningModule):
    def __init__(self, token_vocab_size, docid_vocab_size, d_model=8, nhead=4, num_layers=1, conv_channels=8, kernel_size=3):
        super(Seq2SeqTransformer, self).__init__()

        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.validation_accuracy_outputs = []

        # Embedding layer
        self.embedding = nn.Embedding(token_vocab_size, d_model)

        # Transformer layers
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=128,
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

        # Transformer
        output_transformer = self.transformer(input_embedding, target_embedding)

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
        optimizer = optim.AdamW(self.parameters(), lr=5e-4)
        return optimizer