import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
import torch.nn.functional as F
from collections import deque

class KD_ContrastLoss(nn.Module):
    def __init__(self):
        super(KD_ContrastLoss, self).__init__()
    
    def unique_batch_split(self, positions, batch_size):
        unique_batches, inverse_indices = torch.unique(positions[:, 0], return_inverse=True)
        batch_counts = torch.bincount(inverse_indices)
        splits = torch.split(positions, batch_counts.tolist())
        batch_positions = {key: torch.empty((0,)) for key in range(batch_size)}  # 모든 key를 기본적으로 빈 리스트로 초기화
        batch_positions.update({unique_batches[i].item(): splits[i] for i in range(len(unique_batches))})
        return batch_positions
    
    def forward(self, net_output: torch.Tensor, student_feature: torch.Tensor, teacher_feature: torch.Tensor, target: torch.Tensor, kidney_deque: deque):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        kidney_mask = torch.isin(target, torch.tensor([1],device=net_output.device))
        tumor_mask = torch.isin(target, torch.tensor([2],device=net_output.device))
        batch_size = net_output.size(0)
        
        tumor_pred_mask = torch.isin(torch.argmax(net_output, dim=1, keepdim=True), torch.tensor([1],device=net_output.device))   
        kidney_pred_mask = torch.isin(torch.argmax(net_output, dim=1, keepdim=True), torch.tensor([0],device=net_output.device))   
        
        tumor_wrong_pred_mask = tumor_mask & ~tumor_pred_mask
        
        tumor_mask = tumor_mask & tumor_pred_mask
        kidney_mask = kidney_mask & kidney_pred_mask
        
        kidney_positions = kidney_mask.squeeze(dim=1).nonzero(as_tuple=False)
        tumor_positions = tumor_mask.squeeze(dim=1).nonzero(as_tuple=False)
        tumor_wrong_positions = tumor_wrong_pred_mask.squeeze(dim=1).nonzero(as_tuple=False)
        
        if not len(tumor_wrong_positions) == 0:
            batch_kidney_positions = self.unique_batch_split(kidney_positions, batch_size)
            batch_tumor_positions = self.unique_batch_split(tumor_positions, batch_size)
            batch_target_positions = self.unique_batch_split(tumor_wrong_positions, batch_size)

            # 각 배치별 벡터 저장 리스트
            batch_kidney_vector = []
            batch_tumor_vector = []
            batch_target_vector = []
            check=False
            for batch_num in range(0,batch_size):
                if not len(batch_kidney_positions[batch_num]) == 0:
                    student_kidney_feature = student_feature * kidney_mask
                    kidney_vector = student_kidney_feature[batch_num].mean(dim=(1,2,3))
                    kidney_deque.append(kidney_vector.detach())
                
                if not len(batch_tumor_positions[batch_num]) == 0 and not len(kidney_deque) == 0 and not len(batch_target_positions[batch_num]) == 0:
                    teacher_tumor_feature = teacher_feature * tumor_mask
                    tumor_vector = teacher_tumor_feature[batch_num].mean(dim=(1,2,3)) ## (channel,) 벡터로 변환

                    target_feature = student_feature * tumor_wrong_pred_mask
                    target_vector = target_feature[batch_num].mean(dim=(1,2,3)) ## (channel,) 벡터로 변환
                    
                    tensor_list = list(kidney_deque)  # deque → list 변환
                    stacked_kidney_vector = torch.stack(tensor_list)
                    
                    ### dot product
                    tumor_vector = tumor_vector / (torch.norm(tumor_vector, p=2, dim=-1, keepdim=True) + 1e-8)
                    target_vector = target_vector / (torch.norm(target_vector, p=2, dim=-1, keepdim=True) + 1e-8)
                    stacked_kidney_vector = stacked_kidney_vector / (torch.norm(stacked_kidney_vector, p=2, dim=-1, keepdim=True) + 1e-8)
                    
                    batch_tumor_vector.append(tumor_vector)
                    batch_kidney_vector.append(stacked_kidney_vector)
                    batch_target_vector.append(target_vector)
                    check=True
            if check:
                tumor_similarity = torch.stack([torch.sum(target_v * tumor_v) for tumor_v, target_v in zip(batch_tumor_vector, batch_target_vector)], dim=0)
                kidney_similarity = torch.cat([torch.sum(target_v.unsqueeze(0) * kidney_v, dim=-1) for kidney_v, target_v in zip(batch_kidney_vector, batch_target_vector)], dim=0)
                
                exp_t = torch.exp(tumor_similarity)
                exp_k = torch.exp(kidney_similarity)
            
                sim_loss = -1 /batch_size * torch.log(exp_t.sum() / exp_k.sum())            
            else:
                sim_loss = torch.tensor(0.0, device=net_output.device)  # GPU에서 실행되도록 보장
        else:
            sim_loss = torch.tensor(0.0, device=net_output.device)  # GPU에서 실행되도록 보장
        return sim_loss


class NCCT_ContrastLoss(nn.Module):
    def __init__(self):
        super(NCCT_ContrastLoss, self).__init__()

    def unique_batch_split(self, positions, batch_size):
        unique_batches, inverse_indices = torch.unique(positions[:, 0], return_inverse=True)
        batch_counts = torch.bincount(inverse_indices)
        splits = torch.split(positions, batch_counts.tolist())
        batch_positions = {key: torch.empty((0,)) for key in range(batch_size)}  # 모든 key를 기본적으로 빈 리스트로 초기화
        batch_positions.update({unique_batches[i].item(): splits[i] for i in range(len(unique_batches))})
        return batch_positions

    def forward(self, net_output: torch.Tensor, feature: torch.Tensor, target: torch.Tensor, kidney_deque: deque):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        kidney_mask = torch.isin(target, torch.tensor([1],device=net_output.device))
        tumor_mask = torch.isin(target, torch.tensor([2],device=net_output.device))
        batch_size = net_output.size(0)
        
        tumor_pred_mask = torch.isin(torch.argmax(net_output, dim=1, keepdim=True), torch.tensor([1],device=net_output.device))   
        kidney_pred_mask = torch.isin(torch.argmax(net_output, dim=1, keepdim=True), torch.tensor([0],device=net_output.device))    
        tumor_wrong_pred_mask = tumor_mask & ~tumor_pred_mask
        
        tumor_mask = tumor_mask & tumor_pred_mask
        kidney_mask = kidney_mask & kidney_pred_mask

        kidney_positions = kidney_mask.squeeze(dim=1).nonzero(as_tuple=False)
        tumor_positions = tumor_mask.squeeze(dim=1).nonzero(as_tuple=False)
        tumor_wrong_positions = tumor_wrong_pred_mask.squeeze(dim=1).nonzero(as_tuple=False)
        check=False
        batch_kidney_vector = []
        batch_tumor_vector = []
        batch_target_vector = []
        if not len(tumor_wrong_positions) == 0:
            batch_kidney_positions = self.unique_batch_split(kidney_positions, batch_size)
            batch_tumor_positions = self.unique_batch_split(tumor_positions, batch_size)
            batch_target_positions = self.unique_batch_split(tumor_wrong_positions, batch_size)

            for batch_num in range(0,batch_size):
                if not len(batch_kidney_positions[batch_num]) == 0:
                    student_kidney_feature = feature * kidney_mask
                    kidney_vector = student_kidney_feature[batch_num].mean(dim=(1,2,3))
                    kidney_deque.append(kidney_vector.detach())

                if not len(batch_tumor_positions[batch_num]) == 0 and not len(kidney_deque) == 0 and not len(batch_target_positions[batch_num]) == 0:
                    teacher_tumor_feature = feature * tumor_mask
                    tumor_vector = teacher_tumor_feature[batch_num].mean(dim=(1,2,3)) ## (channel,) 벡터로 변환

                    target_feature = feature * tumor_wrong_pred_mask
                    target_vector = target_feature[batch_num].mean(dim=(1,2,3)) ## (channel,) 벡터로 변환
                    
                    tensor_list = list(kidney_deque)  # deque → list 변환
                    stacked_kidney_vector = torch.stack(tensor_list)
                    
                    ### dot product
                    tumor_vector = tumor_vector / (torch.norm(tumor_vector, p=2, dim=-1, keepdim=True) + 1e-8)
                    target_vector = target_vector / (torch.norm(target_vector, p=2, dim=-1, keepdim=True) + 1e-8)
                    stacked_kidney_vector = stacked_kidney_vector / (torch.norm(stacked_kidney_vector, p=2, dim=-1, keepdim=True) + 1e-8)
                    
                    batch_tumor_vector.append(tumor_vector)
                    batch_kidney_vector.append(stacked_kidney_vector)
                    batch_target_vector.append(target_vector)
                    check=True
                    
                    ## dot product
                    tumor_vector = tumor_vector / (torch.norm(tumor_vector, p=2, dim=-1, keepdim=True) + 1e-8)
                    target_vector = target_vector / (torch.norm(target_vector, p=2, dim=-1, keepdim=True) + 1e-8)
                    stacked_kidney_vector = stacked_kidney_vector / (torch.norm(stacked_kidney_vector, p=2, dim=-1, keepdim=True) + 1e-8)
                    
                    batch_tumor_vector.append(tumor_vector)
                    batch_kidney_vector.append(stacked_kidney_vector)
                    batch_target_vector.append(target_vector)
                    check=True
            if check:
                tumor_similarity = torch.stack([torch.sum(target_v * tumor_v.detach()) for tumor_v, target_v in zip(batch_tumor_vector, batch_target_vector)], dim=0)
                kidney_similarity = torch.cat([torch.sum(target_v.unsqueeze(0) * kidney_v.detach(), dim=-1) for kidney_v, target_v in zip(batch_kidney_vector, batch_target_vector)], dim=0)
                
                exp_t = torch.exp(tumor_similarity)
                exp_k = torch.exp(kidney_similarity)
            
                sim_loss = -1 /batch_size * torch.log(exp_t.sum() / exp_k.sum()) 
            else:
                sim_loss = torch.tensor(0.0, device=net_output.device)  # GPU에서 실행되도록 보장
        else:
            sim_loss = torch.tensor(0.0, device=net_output.device)  # GPU에서 실행되도록 보장
        return sim_loss

class ContrastLoss(nn.Module):
    def __init__(self):
        super(ContrastLoss, self).__init__()

    def unique_batch_split(self, positions, batch_size):
        unique_batches, inverse_indices = torch.unique(positions[:, 0], return_inverse=True)
        batch_counts = torch.bincount(inverse_indices)
        splits = torch.split(positions, batch_counts.tolist())
        batch_positions = {key: torch.empty((0,)) for key in range(batch_size)}  # 모든 key를 기본적으로 빈 리스트로 초기화
        batch_positions.update({unique_batches[i].item(): splits[i] for i in range(len(unique_batches))})
        return batch_positions

    def forward(self, net_output: torch.Tensor, feature: torch.Tensor, target: torch.Tensor, kidney_deque: deque, background_deque: deque):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        background_mask = torch.isin(target, torch.tensor([0],device=net_output.device))
        kidney_mask = torch.isin(target, torch.tensor([1],device=net_output.device))
        tumor_mask = torch.isin(target, torch.tensor([2],device=net_output.device))
        batch_size = net_output.size(0)
        
        tumor_pred_mask = torch.isin(torch.argmax(net_output, dim=1, keepdim=True), torch.tensor([2],device=net_output.device))   
        kidney_pred_mask = torch.isin(torch.argmax(net_output, dim=1, keepdim=True), torch.tensor([1],device=net_output.device))   
        background_pred_mask = torch.isin(torch.argmax(net_output, dim=1, keepdim=True), torch.tensor([0],device=net_output.device))   
        
        tumor_wrong_pred_mask = tumor_mask & ~tumor_pred_mask
        
        tumor_mask = tumor_mask & tumor_pred_mask
        kidney_mask = kidney_mask & kidney_pred_mask
        background_mask = background_mask & background_pred_mask

        kidney_positions = kidney_mask.squeeze(dim=1).nonzero(as_tuple=False)
        tumor_positions = tumor_mask.squeeze(dim=1).nonzero(as_tuple=False)
        background_positions = background_mask.squeeze(dim=1).nonzero(as_tuple=False)
        tumor_wrong_positions = tumor_wrong_pred_mask.squeeze(dim=1).nonzero(as_tuple=False)
        check=False
        if not len(tumor_wrong_positions) == 0:
            batch_background_positions = self.unique_batch_split(background_positions, batch_size)
            batch_kidney_positions = self.unique_batch_split(kidney_positions, batch_size)
            batch_tumor_positions = self.unique_batch_split(tumor_positions, batch_size)
            batch_target_positions = self.unique_batch_split(tumor_wrong_positions, batch_size)

            # 각 배치별 벡터 저장 리스트
            batch_kidney_similarity = []
            batch_tumor_similarity = []
            batch_background_similarity = []
            for batch_num in range(0,batch_size):
                if not len(batch_kidney_positions[batch_num]) == 0:
                    student_kidney_feature = feature * kidney_mask
                    kidney_vector = student_kidney_feature[batch_num].mean(dim=(1,2,3))
                    kidney_deque.append(kidney_vector.detach())
                
                if not len(batch_background_positions[batch_num]) == 0:
                    student_background_feature = feature * background_mask
                    background_vector = student_background_feature[batch_num].mean(dim=(1,2,3))
                    background_deque.append(background_vector.detach())
                
                if not len(batch_tumor_positions[batch_num]) == 0 and not len(kidney_deque) == 0 and not len(background_deque) == 0 and not len(batch_target_positions[batch_num]) == 0:
                    student_tumor_feature = feature * tumor_mask
                    temp_tumor_vector = student_tumor_feature[batch_num].mean(dim=(1,2,3)) ## (channel,) 벡터로 변환
                
                    sample_target_pos = batch_target_positions[batch_num]    
                    target_depth_idx, target_height_idx, target_width_idx = sample_target_pos[:, 1], sample_target_pos[:, 2], sample_target_pos[:, 3]
                        
                    target_vector = feature[batch_num, :, target_depth_idx, target_height_idx, target_width_idx].T
                    tumor_vector = temp_tumor_vector.unsqueeze(0).expand(target_vector.shape[0], -1)
                    
                    tensor_list = list(kidney_deque)  # deque → list 변환
                    stacked_kidney_vector = torch.stack(tensor_list)
                    bg_tensor_list = list(background_deque)  # deque → list 변환
                    stacked_bg_vector = torch.stack(bg_tensor_list)
                    
                    ### dot product
                    tumor_vector = tumor_vector / (torch.norm(tumor_vector, p=2, dim=1, keepdim=True) + 1e-8)
                    target_vector = target_vector / (torch.norm(target_vector, p=2, dim=1, keepdim=True) + 1e-8)
                    stacked_kidney_vector = stacked_kidney_vector / (torch.norm(stacked_kidney_vector, p=2, dim=1, keepdim=True) + 1e-8)
                    
                    batch_tumor_similarity.append(tumor_similarity)
                    batch_kidney_similarity.append(kidney_similarity.reshape(-1))
                    batch_background_similarity.append(bg_similarity.reshape(-1))
                    
                    check=True
            if check:
                tumor_similarity = torch.stack([torch.sum(target_v * tumor_v) for tumor_v, target_v in zip(batch_tumor_vector, batch_target_vector)], dim=0)
                kidney_similarity = torch.cat([torch.sum(target_v.unsqueeze(0) * kidney_v, dim=-1) for kidney_v, target_v in zip(batch_kidney_vector, batch_target_vector)], dim=0)
                
                exp_t = torch.exp(tumor_similarity)
                exp_k = torch.exp(kidney_similarity)
            
                sim_loss = -1 /batch_size * torch.log(exp_t.sum() / exp_k.sum())  
            else:
                sim_loss = torch.tensor(0.0, device=net_output.device)  # GPU에서 실행되도록 보장
        else:
            sim_loss = torch.tensor(0.0, device=net_output.device)  # GPU에서 실행되도록 보장
        return sim_loss