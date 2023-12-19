import torch
import torch.nn as nn
import pytorch_lightning as pl

class SiameseNetwork(pl.LightningModule):
    def __init__(self, input_size, conv_channels):
        super(SiameseNetwork, self).__init__()
        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.validation_accuracy_outputs = []
        dropout_prob = 0.5

        # Siamese network architecture with Convolutional layers
        self.siamese_network = nn.Sequential(
            nn.Conv1d(1, conv_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Calculate the size of the linear layer input after convolutions
        linear_input_size = 2*conv_channels[2] * (input_size)

        self.fc_layers = nn.Sequential(
            nn.Linear(linear_input_size, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),

            nn.Linear(128, 8),
            nn.ReLU(),
        )

        self.fc = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, query, document):
        # Pass query and document through the siamese network
        output_query = self.siamese_network(query.unsqueeze(-1).permute(0,2,1))
        output_document = self.siamese_network(document.unsqueeze(-1).permute(0,2,1))

        # Concatenate the output embeddings along the correct dimension
        combined_embedding = torch.cat((output_query, output_document), dim=1)
        #print(combined_embedding.shape)
        # Pass through the fully connected layers
        relevance_embedding = self.fc_layers(combined_embedding.view(combined_embedding.size(0), -1))

        # Fully connected layer for relevance prediction
        relevance_score = self.fc(relevance_embedding)
        relevance_prob = self.sigmoid(relevance_score)

        return relevance_prob



    def calculate_accuracy(self, predictions, labels):
        predictions = (predictions > 0.5).float()  # Assuming binary classification
        correct = (predictions == labels).float()
        accuracy = correct.sum() / len(correct)
        return accuracy.item()

    def training_step(self, batch, batch_idx):
        query = batch['query']
        document = batch['document']
        relevance = batch['relevance']

        # Forward pass
        similarity = self(query.float(), document.float()).squeeze()
        # Calculate binary cross-entropy loss
        loss = self.criterion(similarity, relevance)
        self.train_step_outputs.append(loss)

        # Log the training loss for tensorboard
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)

        accuracy = self.calculate_accuracy(similarity, relevance)

        return loss

    def validation_step(self, batch, batch_idx):
        query = batch['query']
        document = batch['document']
        relevance = batch['relevance']

        # Forward pass
        similarity = self(query.float(), document.float()).squeeze()
        # Calculate binary cross-entropy loss
        loss = self.criterion(similarity, relevance)
        self.validation_step_outputs.append(loss)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        accuracy = self.calculate_accuracy(similarity, relevance)
        self.validation_accuracy_outputs.append(accuracy)

        return loss

    def on_validation_epoch_end(self):
        if not len(self.train_step_outputs) == 0:
            epoch_average_train = torch.stack(self.train_step_outputs).mean()
            self.log("train_epoch_average", epoch_average_train)
            print("train_loss_avg: ", epoch_average_train)
            self.train_step_outputs.clear()

        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_average", epoch_average)
        print("val_loss_avg: ", epoch_average)
        self.validation_step_outputs.clear()

        accuracy_avg = sum(self.validation_accuracy_outputs) / len(self.validation_accuracy_outputs)
        print("accuracy: ", accuracy_avg)
        self.validation_accuracy_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)