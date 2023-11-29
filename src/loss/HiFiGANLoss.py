import torch
from torch import nn


from src.utils.preprocessing import MelSpectrogram

class MelSpecLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.melspec = MelSpectrogram()
        self.l1_loss = nn.L1Loss()

    def forward(self, 
                real_spec, 
                fake_audio
        ):
        fake_spec = self.melspec(fake_audio)
        return self.l1_loss(real_spec, fake_spec)
  

class FeatureMapLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def _forward(self, real, fake):
        loss = 0
        for r, f in zip(real, fake):
            loss = loss + self.l1_loss(r, f)
        return loss
    
    def forward(self,
                period_real,
                period_fake,
                scale_real,
                scale_fake,
                ):
        period_loss = self._forward(period_real, period_fake)
        scale_loss = self._forward(scale_real, scale_fake)
        return period_loss + scale_loss
    

class GeneratorLoss(nn.Module):
    def __init__(self, lambda_fmap=2, lambda_mel=45):
        super().__init__()
        self.feature_map_loss = FeatureMapLoss()
        self.mel_spec_loss = MelSpecLoss()
        self.lambda_fmap = lambda_fmap
        self.lambda_mel = lambda_mel

    def forward(self,
                period_real,
                period_fake,
                scale_real,
                scale_fake,
                real_spec,
                fake_audio,
                ):
        fmap_loss = self.feature_map_loss(period_real, period_fake, scale_real, scale_fake)
        mel_loss = self.mel_spec_loss(real_spec, fake_audio)

        adv_loss = 0
        for period in period_fake:
            adv_loss = adv_loss + torch.mean((period - 1) ** 2)
        for scale in scale_fake:
            adv_loss = adv_loss + torch.mean((scale - 1) ** 2)

        return fmap_loss * self.lambda_fmap + mel_loss * self.lambda_mel + adv_loss



class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _forward(self, real, fake):
        total_loss = 0
        for r, f in zip(real, fake):
            r_loss = torch.mean((1 - r) ** 2)
            g_loss = torch.mean(f ** 2)
            total_loss = total_loss + (r_loss + g_loss)

        return total_loss

    def forward(self, 
                period_real, 
                period_fake,
                scale_real,
                scale_fake,):
        period_loss = self._forward(period_real, period_fake)
        scale_loss = self._forward(scale_real, scale_fake)
        return period_loss + scale_loss
        