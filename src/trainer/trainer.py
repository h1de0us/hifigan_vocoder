import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
import torchaudio
from tqdm import tqdm

from src.base import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import inf_loop, MetricTracker

from src.utils import ROOT_PATH
from src.utils.preprocessing import MelSpectrogram, MelSpectrogramConfig


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            generator_criterion,
            discriminator_criterion,
            generator_optimizer,
            discriminator_optimizer,
            config,
            device,
            dataloaders,
            generator_lr_scheduler,
            discriminator_lr_scheduler,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, 
                         generator_criterion, 
                         discriminator_criterion, 
                         generator_optimizer, 
                         discriminator_optimizer,
                         generator_lr_scheduler,
                         discriminator_lr_scheduler,
                         config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "generator_loss", 
            "discriminator_loss",
            "feature_map_loss",
            "mel_spectrogram_loss",
            "adversarial_loss",
            "generator grad norm",
            "discriminator grad norm",
            writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "audio"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        # DEBUG
        # torch.autograd.set_detect_anomaly(True)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("generator grad norm", self.get_grad_norm(type="generator"))
            self.train_metrics.update("discriminator grad norm", self.get_grad_norm(type="discriminator"))
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Generator loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["generator_loss"].item()
                    )
                )
                self.logger.debug(
                    "Train Epoch: {} {} Discriminator loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["discriminator_loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "generator learning rate", self.generator_lr_scheduler.get_last_lr()[0]
                )
                self.writer.add_scalar(
                    "discriminator learning rate", self.discriminator_lr_scheduler.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        if self.generator_lr_scheduler is not None:
            self.generator_lr_scheduler.step()
        if self.discriminator_lr_scheduler is not None:
            self.discriminator_lr_scheduler.step()

        # TODO: call _evaluation_epoch here
        self._evaluation_epoch(epoch)

        # self._log_audio(name='train.wav', audio=batch["generated_audio"][0], sample_rate=22050)

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        generated_audio = self.model(**batch)
        if type(generated_audio) is dict:
            batch.update(generated_audio)
        else:
            batch["generated_audio"] = generated_audio

        # detach to properly calculate generator loss
        scales_real, scales_fake, \
               periods_real, periods_fake, \
               feature_maps_real_s, feature_maps_fake_s, \
               feature_maps_real_p, feature_maps_fake_p = self.model.discriminate(real=batch["audio"],
                                                                                  fake=batch["generated_audio"].detach())

        if not is_train:
            return batch
        
        # in training
        self.discriminator_optimizer.zero_grad()
        discriminator_loss = self.discriminator_criterion(periods_real,
                                                          periods_fake,
                                                          scales_real,
                                                          scales_fake)
        discriminator_loss.backward()
        self.discriminator_optimizer.step()
        batch["discriminator_loss"] = discriminator_loss
        self.train_metrics.update("discriminator_loss", discriminator_loss.item())

        # second discriminate to update values
        # here we do not use detach
        scales_real, scales_fake, \
               periods_real, periods_fake, \
               feature_maps_real_s, feature_maps_fake_s, \
               feature_maps_real_p, feature_maps_fake_p = self.model.discriminate(real=batch["audio"],
                                                                                  fake=batch["generated_audio"])


        self.generator_optimizer.zero_grad()
        generator_loss, fmap_loss, mel_loss, adv_loss = self.generator_criterion(periods_fake,
                                                                                scales_fake,
                                                                                feature_maps_real_s,
                                                                                feature_maps_fake_s,
                                                                                feature_maps_real_p,
                                                                                feature_maps_fake_p,
                                                                                batch["spectrogram"],
                                                                                batch["generated_audio"],)
        generator_loss.backward()
        self.generator_optimizer.step()
        batch["generator_loss"] = generator_loss
        batch["feature_map_loss"] = fmap_loss
        batch["mel_spectrogram_loss"] = mel_loss
        batch["adversarial_loss"] = adv_loss

        self.train_metrics.update("generator_loss", generator_loss.item())  
        self.train_metrics.update("feature_map_loss", fmap_loss.item())
        self.train_metrics.update("mel_spectrogram_loss", mel_loss.item())
        self.train_metrics.update("adversarial_loss", adv_loss.item())

        self._clip_grad_norm()

        return batch

    def _evaluation_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """

        melspec = MelSpectrogram(MelSpectrogramConfig)
        mels = []
        for i in range(1, 4):
            path = ROOT_PATH / 'test_data' / f'audio_{i}.wav'
            audio = self._load_audio(path).detach().cpu()
            mels.append(melspec(audio))

        with torch.no_grad():
            self.model.eval()
            save = ROOT_PATH / 'results'
            save.mkdir(exist_ok=True, parents=True)

            for i, mel in enumerate(mels):
                audio = self.model(mel.to(self.device)).squeeze(0)
                self._log_audio(f'audio_{i}', audio, 22050)

        
        

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2, type: str = "generator"):
        if type == "generator":
            parameters = self.model.generator.parameters()
        elif type == "discriminator":
            parameters = self.model.discriminator.parameters()
        else:
            raise NotImplementedError
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    @staticmethod
    def _log_audio(self, name, audio, sample_rate):
        if self.writer is None:
            return
        self.writer.add_audio(name, audio, sample_rate)

    @staticmethod
    def _load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = 22050
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor
