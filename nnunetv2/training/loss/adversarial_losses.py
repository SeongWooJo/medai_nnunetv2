import torch
import torch.nn as nn
import numpy as np

class adversarial_losses:
    @staticmethod
    def ZeroCenteredGradientPenalty(Samples, Critics):
        Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
        return Gradient.square().sum([1, 2, 3])

    def AccumulateGeneratorGradients(NCCTLogits, CECTLogits):
        # Generator 입장, 즉 여기서는 Encoder의 입장에선 CECT - NCCT Logits값을 작게 만드는 방향으로 학습 -> NCCT Logits을 키우는 방향으로 학습을 수행
        RelativisticLogits = NCCTLogits - CECTLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
        
        GeneratorLoss = AdversarialLoss.mean()
        
        return GeneratorLoss, RelativisticLogits

    def SimpleDiscriminatorGradients(NCCTLogits, CECTLogits):
        RelativisticLogits = CECTLogits - NCCTLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
        DiscriminatorLoss = AdversarialLoss.mean()
        return DiscriminatorLoss, RelativisticLogits

    def AccumulateDiscriminatorGradients(NCCTSamples, CECTSamples, NCCTLogits, CECTLogits, Gamma):
        R1Penalty = adversarial_losses.ZeroCenteredGradientPenalty(CECTSamples, CECTLogits)
        R2Penalty = adversarial_losses.ZeroCenteredGradientPenalty(NCCTSamples, NCCTLogits)
        
        # Loss가 줄어들기 위해선 NCCT - CECT LOGITS이 값이 작아질질수록 Loss가 준다. 즉, CECT Logits을 크게 만들게 학습을 수행
        RelativisticLogits = CECTLogits - NCCTLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
        
        temp_DiscriminatorLoss = AdversarialLoss + (Gamma / 2) * (R1Penalty + R2Penalty)
        DiscriminatorLoss = temp_DiscriminatorLoss.mean()
        
        return DiscriminatorLoss, RelativisticLogits, R1Penalty, R2Penalty
    
    def cosine_decay_with_warmup(current_epoch, base_value, max_epoch, final_value=0.0, warmup_value=0.0, warmup_nimg=0, hold_base_value_nimg=0):
        decay = 0.5 * (1 + np.cos(np.pi * (current_epoch - warmup_nimg - hold_base_value_nimg) / float(max_epoch - warmup_nimg - hold_base_value_nimg)))
        cur_value = base_value + (1 - decay) * (final_value - base_value)
        if hold_base_value_nimg > 0:
            cur_value = np.where(current_epoch > warmup_nimg + hold_base_value_nimg, cur_value, base_value)
        if warmup_nimg > 0:
            slope = (base_value - warmup_value) / warmup_nimg
            warmup_v = slope * current_epoch + warmup_value
            cur_value = np.where(current_epoch < warmup_nimg, warmup_v, cur_value)
        return float(np.where(current_epoch > max_epoch, final_value, cur_value))