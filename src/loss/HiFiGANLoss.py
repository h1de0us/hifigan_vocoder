import torch
from torch import nn


from src.utils.preprocessing import MelSpectrogram, MelSpectrogramConfig

class MelSpecLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.melspec = MelSpectrogram(MelSpectrogramConfig())
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
            for rmap, fmap in zip(r, f):
                rmap, fmap = torch.tensor(rmap), torch.tensor(fmap)
                loss = loss + self.l1_loss(rmap, fmap)
        return loss
    
    def forward(self,
                feature_maps_real_s,
                feature_maps_fake_s,
                feature_maps_real_p,
                feature_maps_fake_p,
                ):
        period_loss = self._forward(feature_maps_real_p, feature_maps_fake_p)
        scale_loss = self._forward(feature_maps_real_s, feature_maps_fake_s)
        return period_loss + scale_loss
    

class GeneratorLoss(nn.Module):
    def __init__(self, lambda_fmap=2, lambda_mel=45):
        super().__init__()
        self.feature_map_loss = FeatureMapLoss()
        self.mel_spec_loss = MelSpecLoss()
        self.lambda_fmap = lambda_fmap
        self.lambda_mel = lambda_mel

    def forward(self,
                period_fake,
                scale_fake,
                feature_maps_real_s,
                feature_maps_fake_s,
                feature_maps_real_p,
                feature_maps_fake_p,
                real_spec,
                fake_audio,
                ):
        fmap_loss = self.feature_map_loss(feature_maps_real_s,
                                            feature_maps_fake_s,
                                            feature_maps_real_p,
                                            feature_maps_fake_p,)
        mel_loss = self.mel_spec_loss(real_spec, fake_audio)

        adv_loss = 0
        for period in period_fake:
            adv_loss = adv_loss + torch.mean((period - 1) ** 2)
        for scale in scale_fake:
            adv_loss = adv_loss + torch.mean((scale - 1) ** 2)

        generator_loss = fmap_loss * self.lambda_fmap + mel_loss * self.lambda_mel + adv_loss
        return generator_loss, fmap_loss, mel_loss, adv_loss



class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _forward(self, real, fake):
        total_loss = 0
        for r, f in zip(real, fake):
            r_loss = torch.mean((1 - r) ** 2)
            g_loss = torch.mean(f ** 2)
            total_loss = total_loss + r_loss
            total_loss = total_loss + g_loss

        return total_loss

    def forward(self, 
                period_real, 
                period_fake,
                scale_real,
                scale_fake,):
        period_loss = self._forward(period_real, period_fake)
        scale_loss = self._forward(scale_real, scale_fake)
        return period_loss + scale_loss
        