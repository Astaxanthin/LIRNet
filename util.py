import os
import torch as t
import numpy as np
import cv2
import copy


from loss import strongLossFunction

# forward validate
def validate(dataloader, net, device, epoch, patch_size, num_data,batch_size,r_thd):
    
    net.eval()

    loss_func = strongLossFunction()
    running_loss = 0.0
    ae = 0
    se = 0
    count_pre = 0
    all_density_map = []
    all_location_gt = []
    with t.no_grad():
        for data in dataloader:

            inputs,labels_dot, labels_mask = data[0].to(device),data[1].to(device),data[2].to(device)

            outputs_density = net(inputs)
            all_density_map.append(outputs_density.squeeze().cpu().numpy())
            all_location_gt.append(labels_dot.squeeze().cpu().numpy())

            save_visualization(outputs_density[0,0,:,:].squeeze().cpu().numpy(), epoch)
            
            gt = t.sum(labels_dot,[2,3])
            ae += t.abs(t.sum(outputs_density,[2,3])-gt).sum()
            se += ((t.sum(outputs_density,[2,3])-gt)*(t.sum(outputs_density,[2,3])-gt)).sum()
            count_pre += t.sum(outputs_density,[2,3]).sum()

            loss = loss_func(outputs_density,labels_dot.float(),labels_mask.float(),patch_size)
            running_loss += loss.item()*batch_size
        
        val_loss = running_loss/num_data
        val_mae = ae/num_data
        val_mse = t.sqrt(se/num_data)
        count_mean = count_pre/num_data

        all_density_map = np.array(all_density_map)
        all_location_gt = np.array(all_location_gt)
        all_maps = all_density_map[0]
        all_labels = all_location_gt[0]
        for j in range(len(all_density_map)-1):
            all_maps = np.append(all_maps,all_density_map[j+1],axis = 0)
            all_labels = np.append(all_labels, all_location_gt[j+1], axis = 0)

        print(all_maps.shape)
        val_F1 = location_eval(all_maps, all_labels, r_thd)

    return val_loss, count_mean, val_mae, val_mse, val_F1

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()

def save_visualization(density, epoch):
    density_img = density/max([density.max(),0.0001])*255
    density_img = np.array(density_img,dtype = np.uint8)
    save_dir = './result/density_imgs/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    cv2.imwrite('./result/density_imgs/density_' + str(epoch) + '.bmp', density_img)

# forward test
def test_full_img(dataloader,model,device,save_address,img_name_list):
    
    model.eval()
    ae = 0
    se = 0
    count_pre = 0
    idex = 0
    with t.no_grad():
        for data in dataloader:
            
            inputs,dot_map = data[0].to(device),data[1].to(device)

            outputs_map = model(inputs)

            gt = t.sum(dot_map,[2,3])
            ae += t.abs(t.sum(outputs_map,[2,3])-gt).sum()
            se += ((t.sum(outputs_map,[2,3])-gt)*(t.sum(outputs_map,[2,3])-gt)).sum()
            count_pre += t.sum(outputs_map,[2,3]).sum()
            
            outputs_map = outputs_map.squeeze().cpu().numpy()
            ###********************************************### 
            print(f'{idex}, {img_name_list[idex]}, {outputs_map.shape}')
            np.savetxt(f'{save_address}/{img_name_list[idex]}_localization.txt',outputs_map,fmt='%0.4f')
            idex += 1

        
        MAE = ae/(len(img_name_list))
        MSE = t.sqrt(se/(len(img_name_list)))
        count_mean = count_pre/(len(img_name_list))

        print(f'MAE: %.2f, MSE: %.2f, Count_mean: %.2f'%(MAE, MSE, count_mean))


def location(density, r_thd):
    
    count = density.sum()
    count_int = int(round(count))
    result = []
    for i in range(count_int):

        index = np.where(density == density.max())
        result.append([index[1][0],index[0][0]])
        cv2.circle(density,(index[1][0],index[0][0]), r_thd, 0,-1) ## NMS

    return result

def distance_sq(gt_pos,pre_pos):
    
    # gt_pos  2*M  
    # pre_pos 2*N
    # output  M*N

    N_gt = gt_pos.shape[1]
    N_pre = pre_pos.shape[1]
    gt2 = (gt_pos**2).sum(0)
    gt2 = gt2[np.newaxis,:]
    pre2 = (pre_pos**2).sum(0)
    pre2 = pre2[np.newaxis,:]

    X2 = np.kron(gt2.T,np.ones([1,N_pre]))
    Y2 = np.kron(pre2,np.ones([N_gt,1]))
    _2XY = 2*np.dot(gt_pos.T,pre_pos)
    dis_mat = X2 + Y2 - _2XY

    return dis_mat

def location_eval(outputs_density, dot_labels, r_thd):
    
    TP_seq = []
    FP_seq = []
    TN_seq = []
    dis_seq = []
    count_pre = []
    count_gt = []

    batchsize = outputs_density.shape[0]
    for i in range(batchsize):
        density = outputs_density[i,:,:]
        dot_label = dot_labels[i,:,:]

        count_pre.append(density.sum())
        count_gt.append(dot_label.sum())

        pre_pos = np.array(location(density, r_thd))             #N*2
        gt_location = np.where(dot_label==1)
        gt_pos = np.vstack((gt_location[1],gt_location[0])).T   #N*2

        ori_pre_pos = copy.deepcopy(pre_pos)

        if min(pre_pos.shape) == 0:
            TP_seq.append(0)
            FP_seq.append(0)
            TN_seq.append(gt_pos.shape[0])
            continue
        if min(gt_pos.shape) == 0:
            TP_seq.append(0)
            FP_seq.append(pre_pos.shape[0])
            TN_seq.append(0)
            continue

        all_dis_mat = np.sqrt(distance_sq(pre_pos.T,gt_pos.T))
        dis_seq.extend(all_dis_mat.min(1))

        TP = 0
        for i in range(gt_pos.shape[0]):
            gt_loc = np.array([gt_pos[i,:]])
            if pre_pos.shape[0] == 0:
                continue
            dis_mat = np.sqrt(distance_sq(gt_loc.T,pre_pos.T))
            
            if dis_mat.min() <= r_thd:
                TP += 1
                idex = np.where(dis_mat == dis_mat.min())
                pre_pos =  np.delete(pre_pos,idex[1],0)
                # dis_seq.append(dis_mat.min())
        

        density = density/max([density.max(),0.0001])*255
        density = np.array(density,dtype = np.uint8)

        TP_seq.append(TP)
        TN = gt_pos.shape[0] - TP
        TN_seq.append(TN)
        FP = ori_pre_pos.shape[0] - TP
        FP_seq.append(FP)


    count_pre = np.array(count_pre)
    count_gt = np.array(count_gt)

    me = np.median(dis_seq)  # median
    q1 = np.percentile(dis_seq,25) # Q1
    q3 = np.percentile(dis_seq,75) # Q3

    mae = (abs(count_pre-count_gt)).mean()
    mse = np.sqrt(((count_pre-count_gt)**2).mean())

    total_tp = np.array(TP_seq).sum()
    total_tn = np.array(TN_seq).sum()
    total_fp = np.array(FP_seq).sum()
    total_p = total_tp + total_fp
    if total_p == 0:
        total_p += 0.01
    total_precision  = total_tp / total_p
    total_recall = total_tp / (total_tp + total_tn)
    
    to = total_precision + total_recall
    if to == 0:
        to += 0.01 
    F1 = 2*total_precision*total_recall/to

    eval_res = {}
    eval_res['mae'] = mae
    eval_res['mse'] = mse
    eval_res['me'] = me
    eval_res['q1'] = q1
    eval_res['q3'] = q3
    eval_res['precision'] = total_precision
    eval_res['recall'] = total_recall
    eval_res['F1'] = F1

    return eval_res