import argparse
import collections
import warnings

import numpy as np
import torch

import src.loss as module_loss
import src.model as module_arch
from src.trainer import Trainer
from src.utils import prepare_device
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # text_encoder

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    generator_loss_module = config.init_obj(config["generator_loss"], module_loss).to(device)
    discriminator_loss_module = config.init_obj(config["discriminator_loss"], module_loss).to(device)

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    generator_optimizer = config.init_obj(config["generator_optimizer"], torch.optim, trainable_params)
    discriminator_optimizer = config.init_obj(config["discriminator_optimizer"], torch.optim, trainable_params)
    generator_lr_scheduler = config.init_obj(config["generator_lr_scheduler"], torch.optim.lr_scheduler, generator_optimizer)
    discriminator_lr_scheduler = config.init_obj(config["discriminator_lr_scheduler"], torch.optim.lr_scheduler, discriminator_optimizer)

    trainer = Trainer(
        model,
        generator_loss_module,
        discriminator_loss_module,
        generator_optimizer,
        discriminator_optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        generator_lr_scheduler=generator_lr_scheduler,
        discriminator_lr_scheduler=discriminator_lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--gen_lr", "--generator_learning_rate"], type=float, target="generator_optimizer;args;lr"),
        CustomArgs(["--dis_lr", "--discriminator_learning_rate"], type=float, target="discriminator_optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
