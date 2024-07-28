from typing import Optional

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision import transforms

from slot_attention.data import CLEVRDataModule
from slot_attention.data import CLEVERERDataModule
from slot_attention.method import SlotAttentionMethod
from slot_attention.model import SlotAttentionModel
from slot_attention.params import SlotAttentionParams
from slot_attention.utils import ImageLogCallback
from slot_attention.utils import rescale


def run_inference(params: Optional[SlotAttentionParams] = None):
    if params is None:
        params = SlotAttentionParams()

    clevr_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(rescale),  # rescale between -1 and 1
            transforms.Resize(params.resolution),
        ]
    )

    clevr_datamodule = CLEVERERDataModule(
        data_root=params.data_root,
        max_n_objects=params.num_slots - 1,
        train_batch_size=params.batch_size,
        val_batch_size=32,
        clevr_transforms=clevr_transforms,
        num_train_images=params.num_train_images,
        num_val_images=256,
        num_workers=params.num_workers,
    )

    model = SlotAttentionModel(
        resolution=params.resolution,
        num_slots=params.num_slots,
        num_iterations=params.num_iterations,
        empty_cache=params.empty_cache,
    )
    
    method = SlotAttentionMethod.load_from_checkpoint(
        checkpoint_path='/home/jovyan/wandb/run-20210720_134256-3qnmcevx/files/slot-attention-clevrer/3qnmcevx/checkpoints/epoch=55-step=69999.ckpt',
        map_location=None,
        model=model,
        datamodule=clevr_datamodule,
        params=params
    )

    trainer = Trainer(
        logger=pl_loggers.CSVLogger("/home/jovyan/test_outputs"),
        accelerator="ddp" if params.gpus > 1 else None,
        gpus=params.gpus,
        # precision=16,
    )
    # trainer.fit(method, clevr_datamodule.train_dataloader(), clevr_datamodule.validation_dataloader())
    trainer.test(method, clevr_datamodule.validation_dataloader())


if __name__ == "__main__":
    main()
