import lightning as L
import torch
import torch.nn as nn


class LinearModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config["model_hparams"]
        self.feats_dim = self.config["feats_dim"]
        self.input_type = self.config["input_type"]

        self.save_hyperparameters()

        self.head = nn.Linear(self.feats_dim, 1)

    def forward(self, input_feats):
        video_feats, audio_feats = input_feats[0], input_feats[1]

        if self.input_type == "audio":
            fused_features = audio_feats
        elif self.input_type == "video" or self.input_type == "multimodal":  # for multimodal, video_feats and audio_feats are equal
            fused_features = video_feats
        else:
            raise ValueError(f"Error! Input type: {self.input_type}")

        output = self.head(fused_features)[:, :, 0]

        return torch.logsumexp(output, dim=-1)

    def predict_scores(self, video_feats, audio_feats):
        scores = self.forward((video_feats, audio_feats))
        return scores

    def training_step(self, batch, batch_idx):
        video_feats, audio_feats, labels, _ = batch
        if video_feats.numel() == 0:
            return

        output = self.forward((video_feats, audio_feats))
        score = output.unsqueeze(1)
        score = torch.cat((-score, score), 1)

        loss = torch.nn.functional.cross_entropy(score, labels)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        video_feats, audio_feats, labels, _ = batch
        if video_feats.numel() == 0:
            return

        output = self.forward((video_feats, audio_feats))
        score = output.unsqueeze(1)
        score = torch.cat((-score, score), 1)

        loss = torch.nn.functional.cross_entropy(score, labels)
        self.log("val_loss", loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return [optimizer], []
