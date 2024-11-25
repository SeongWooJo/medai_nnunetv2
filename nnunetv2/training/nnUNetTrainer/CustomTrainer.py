# MyCustomTrainer.py 파일로 저장
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch.amp import GradScaler

import torch

class CustomTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # 원하는 epoch 수로 변경
        self.num_epochs = 1000  # 예: 2000 epoch으로 설정
        self.grad_scaler = GradScaler() #if self.device.type == 'cuda' else None
        # 학습률을 변경 (예: 0.0005)
        # self.initial_lr = 5e-4
    
    def run_training(self):
        print("Custom Trainer: Starting Training with Custom Settings : New Version!")
        # Custom 학습 루틴 추가
        return super().run_training()
