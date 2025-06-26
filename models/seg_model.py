import torch
from torch.optim import lr_scheduler
from torchmetrics import MeanMetric, MinMetric

import lightning as L
import segmentation_models_pytorch as smp

SEED = 59
L.seed_everything(59)

DATA_DIR = "C:\Users\Admin\Desktop\FINAL-IMAGE-PROCESSING\deep-learning-for-images\data\medical-image-segmentation"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MedSegLitModule(L.LightningModule):
    def __init__(
        self, arch: str = "Unet", encoder_name: str = "resnet34", 
        encoder_weights: str = "imagenet", in_channels: int = 3, 
        out_classes: int = 1, learning_rate: float = 2e-4,
        t_max: int = 55 * 20, # num_epochs * len(train_dataloader)
    ):
        # https://smp.readthedocs.io/en/latest/models.html
        # https://smp.readthedocs.io/en/latest/encoders.html
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
        )

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        # self.loss_fn_2 = smp.losses.SoftBCEWithLogitsLoss()
        # self.loss_fn = smp.losses.MCCLoss()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_loss_best = MinMetric()

        # initialize step metrics lists
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask
    
    def on_train_start(self):
        # Reset metrics at start of training
        self.train_loss.reset()
        self.val_loss.reset()
        self.val_loss_best.reset()
    
    def shared_step(self, batch, stage):
        image = batch["image"]
        mask = batch["mask"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)
        # loss = self.loss_fn(logits_mask, mask) + self.loss_fn_2(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        if stage == "train":
            metrics[f"{stage}_loss"] = self.train_loss.compute()
        elif stage == "valid":
            self.val_loss_best(self.val_loss.compute())
            metrics[f"{stage}_loss"] = self.val_loss.compute()
            metrics[f"{stage}_loss_best"] = self.val_loss_best.compute()

        self.log_dict(metrics, prog_bar=True)

        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Epoch {self.current_epoch}: {metrics_str}")

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.train_loss(train_loss_info["loss"])
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # empty set output list
        self.train_loss.reset()
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.val_loss(valid_loss_info["loss"])
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.log("val/loss_best", self.val_loss_best.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.val_loss.reset()
        self.validation_step_outputs.clear()
        return
    
    def test_step(self, batch, batch_idx):
        pass

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.t_max, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }