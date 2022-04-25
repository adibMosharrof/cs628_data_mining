from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.utilities.cli import (DATAMODULE_REGISTRY,
                                             MODEL_REGISTRY,
                                             LightningArgumentParser,
                                             LightningCLI)

import data_modules
import embedding_utils
import models

MODEL_REGISTRY.register_classes(models, LightningModule)
DATAMODULE_REGISTRY.register_classes(data_modules, LightningDataModule)


class MyLightningCLI(LightningCLI):
    # def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
    # super().add_arguments_to_parser(parser)
    # parser.add_argument("--plot_name", default="plot_embeddings")

    def after_fit(self):
        (train_embeddings, train_labels,) = embedding_utils.extract_embeddings(
            self.datamodule.get_base_train_dataloader(), self.model
        )
        embedding_utils.plot_embeddings(
            train_embeddings,
            train_labels,
            out_dir=self.trainer.log_dir,
            plot_file_name="train_embeddings",
        )

        (val_embeddings, val_labels,) = embedding_utils.extract_embeddings(
            self.datamodule.get_base_val_dataloader(), self.model
        )
        embedding_utils.plot_embeddings(
            val_embeddings,
            val_labels,
            out_dir=self.trainer.log_dir,
            plot_file_name="val_embeddings",
        )

    def before_fit(self):
        data_params = vars(self.config.fit.data.init_args)
        model_params = vars(self.config.fit.model.init_args)
        trainer_params = vars(self.config.fit.trainer)
        hparams_dict = {
            "dataset_name": data_params["dataset_name"],
            "metric_name": data_params["metric_name"],
            "loss": model_params["loss_func"]["class_path"],
            "epochs": trainer_params["max_epochs"],
        }

        self.trainer.logger.log_hyperparams(hparams_dict)


if __name__ == "__main__":

    cli = MyLightningCLI()
