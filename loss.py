import torch
import torch.nn as nn
import torch.nn.functional as F

class fullLossFunction(torch.nn.Module):

    def __init__(self):
        super(fullLossFunction, self).__init__()

    def objectloss_func(self,d,label_dot,patch_size):

        output_count = count_data(d,patch_size,1)
        object_loss = torch.abs(output_count-1)
        object_loss = (object_loss.mul(label_dot)).sum(1).sum(1).sum(1)

        return object_loss

    def backloss_func(self, d, label_mask,patch_size):
        back_loss = d.mul(label_mask).sum(1).sum(1).sum(1)
        back_loss = torch.abs(back_loss)

        return back_loss

    def forward(self, d_cn0, label_dot, label_mask, patch_size_pair):

        object_loss = self.objectloss_func(d_cn0, label_dot, patch_size_pair[0])
        back_loss = self.backloss_func(d_cn0, label_mask, patch_size_pair[0])
        
        loss = object_loss + back_loss
        loss = loss*label_dot.sum(1).sum(1).sum(1)/label_dot.sum()
        loss = loss.mean()

        return loss


class weakLossFunction(torch.nn.Module):
    
    def __init__(self, weight=1.0):
        super(weakLossFunction, self).__init__()
        self.weight = weight
    
    def countingloss_func(self,d,label,patch_size):

        masker = maskGenerator(label)
        label_0_mask = masker.generate_label0_mask()
        label_1_mask = masker.generate_label1_mask()
        label_2_mask = masker.generate_label2_mask()

        output_count = count_data(d,patch_size)

        # Truncate the sum to approach 2 by leaky relu 
        counting_pre = 2 - F.leaky_relu(2-output_count,negative_slope = 0.0001)  

        loss_label_0 = label_0_mask.mul(torch.abs(counting_pre)).sum()

        loss_label_1 = label_1_mask.mul(torch.abs(counting_pre-label)).sum()
        
        loss_label_2 = label_2_mask.mul(2 - counting_pre).sum()
        
        return loss_label_0, loss_label_1, loss_label_2

    def maxloss_func(self, d0):
        
        max_pool = nn.MaxPool2d(kernel_size=(d0.shape[2],d0.shape[3]))
        max_d0 = max_pool(d0)
        loss = F.relu(max_d0-1).sum()

        return loss

    def forward(self, d_cn0, large_label, small_label, patch_size_pair):

        large_ct_loss_0, large_ct_loss_1, large_ct_loss_2 = self.countingloss_func(d_cn0, large_label, patch_size_pair[0])
        small_ct_loss_0, small_ct_loss_1, small_ct_loss_2 = self.countingloss_func(d_cn0, small_label, patch_size_pair[1])

        large_masker = maskGenerator(large_label)
        large_0_num = large_masker.generate_label0_mask().sum()
        large_1_num = large_masker.generate_label1_mask().sum()
        large_2_num = large_masker.generate_label2_mask().sum()

        small_masker = maskGenerator(small_label)
        small_0_num = small_masker.generate_label0_mask().sum()
        small_1_num = small_masker.generate_label1_mask().sum()
        small_2_num = small_masker.generate_label2_mask().sum()

        ct_loss_0 = (large_ct_loss_0 + small_ct_loss_0)/max(0.01,large_0_num + small_0_num)
        ct_loss_1 = (large_ct_loss_1 + small_ct_loss_1)/max(0.01,large_1_num + small_1_num)
        ct_loss_2 = (large_ct_loss_2 + small_ct_loss_2)/max(0.01,large_2_num + small_2_num)

        ct_loss = (ct_loss_0 + ct_loss_1 + ct_loss_2)/3

        max_loss = self.maxloss_func(d_cn0)
        loss = ct_loss + 0.001*max_loss
        #print(label_2_numbers) 
        print('loss_count:loss_max = [%.4f, %.4f]'%(ct_loss, max_loss))
        
        return loss


def count_data(d, patch_size, step_size = None):
    
    if step_size is None:
        step_size = patch_size
    else:
        d = F.pad(d,[patch_size//2,patch_size//2,patch_size//2,patch_size//2])
    
    avg_pool = torch.nn.AvgPool2d(patch_size,step_size)

    return patch_size*patch_size*avg_pool(d)

class maskGenerator():
        
    def __init__(self,label):

        self.label = label

    def generate_nolabel_mask(self):
        mask = torch.ones(self.label.shape)
        mask[self.label > -1] = 0
        return mask.cuda()

    def generate_label_mask(self):
        mask = torch.ones(self.label.shape)
        mask[self.label < 0] = 0
        return mask.cuda()

    def generate_label1_mask(self): 
        mask = torch.zeros(self.label.shape)
        mask[self.label == 1.0] = 1.0
        return mask.cuda()
 
    def generate_label0_mask(self): 
        mask = torch.zeros(self.label.shape)
        mask[self.label == 0.0] = 1.0
        return mask.cuda()

    def generate_label2_mask(self):
        mask = torch.zeros(self.label.shape)
        mask[self.label == 2.0] = 1.0
        return mask.cuda()
