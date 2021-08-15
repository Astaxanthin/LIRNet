import torch
from torch.utils import data
from PIL import Image
import numpy as np
import cv2
import os
import glob
import random


class imgDataset(data.Dataset):

    def __init__(self, img_path, label_dot_path, patch_size, transform, train = True, weakly_supervised = False, weak_param = {},random_seed = None):
        
        self.img_list = glob.glob(img_path + '*[jpg,bmp,png,pgm]')
        #print('self.list')
        self.label_dot_path = label_dot_path
        self.kernel_size = patch_size[0]
        self.transform = transform
        self.train = train
        self.weakly = weakly_supervised
        self.weak_param = weak_param
        self.rand_seed = random_seed

    def __getitem__(self, index):
        if self.img_list is None:
            print('Noooooooooooooooooooooooooooo images are found!')
            return None,None
        
        # get img dir
        img_dir = self.img_list[index]
         # get dot label
        img_name = img_dir.split('/')[-1].split('.')[0] 
        label_dot_dir = self.label_dot_path + img_name + '.txt'
        #print(label_dot_dir)

        if os.path.exists(label_dot_dir) is False:
            print('Noooooooooooooooooooooooooooo labels are found!')
            return None,None

        src_img = cv2.imread(self.img_list[index])
        src_img = src_img.astype(np.float32, copy=False)
                
        label = np.loadtxt(label_dot_dir)
        label_object, label_background = get_obejct_background(src_img, label, self.kernel_size)

        if self.train:
            transform_modes = ['unchanged','v','h','vh','r9','r27','r9h','r9v'] # for square image
            #transform_modes = ['unchanged','v','h','vh']  # for rectange image

            rand_transform_mode = transform_modes[np.random.permutation(len(transform_modes))[0]]
            
            rand_transform = RandomTransform()

            inputs = {}
            inputs['transform_mode'] = rand_transform_mode

            inputs['x'] = src_img
            data = self.transform(rand_transform(inputs))

            if not self.weakly:
               
                inputs['x'] = label_object
                label_object = torch.from_numpy(rand_transform(inputs)).unsqueeze(0)
                inputs['x'] = label_background
                label_background = torch.from_numpy(rand_transform(inputs)).unsqueeze(0)

                return data,label_object,label_background
            else :
                if len(self.weak_param) == 0:
                    print('Noooooooooooooooooooooooooooo parameters for waekly supervised algorithm are found!')
                    return None,None

                ## Generate weak supervision: patch-level labels 
                label_img_tensor = torch.from_numpy(label_object).unsqueeze(0).unsqueeze(0).float()
                lg_size = self.weak_param['large_size']
                avg_pool_lg = torch.nn.AvgPool2d(lg_size,lg_size)
                label_lg = lg_size*lg_size*avg_pool_lg(label_img_tensor).squeeze().numpy()
                label_lg[label_lg>1] = 2

                label_lg_mask = np.kron(label_lg>1,np.ones([lg_size,lg_size]))
                label_lg_mask_tentor = torch.from_numpy(label_lg_mask).unsqueeze(0).unsqueeze(0).float()

                sg_size = self.weak_param['small_size']
                avg_pool_sg = torch.nn.AvgPool2d(sg_size,sg_size)
                mask_sg = avg_pool_sg(label_lg_mask_tentor).squeeze().numpy()
                sg_all_num = (mask_sg>0.5).sum()
                # print('all_num: %d'%(sg_all_num))

                small_patch_num_per_large = self.weak_param['large_size']/self.weak_param['small_size']
                small_patch_num_per_large = small_patch_num_per_large*small_patch_num_per_large

                label_sg = sg_size*sg_size*avg_pool_sg(label_img_tensor).squeeze().numpy()
                label_sg[label_sg>1] = 2
                label_sg[mask_sg<=0] = -1
                
                if sg_all_num != 0:

                    large_2 = np.where(label_lg > 1)
                    
                    num_small = int(min(self.weak_param['small_label_num'],len(large_2[0])*small_patch_num_per_large))

                    random.seed(self.rand_seed)  ## Keep the same random patches for every epoch 
                    rand_id = random.sample(range(sg_all_num),num_small)

                    # print((label_sg>-1).sum())

                    id_count = 0
                    rows,cols = label_sg.shape
                    for i in range(rows):
                        for j in range(cols):
                            if label_sg[i][j] == -1:
                                continue
                            if id_count not in rand_id:
                                label_sg[i][j] = -1
                            id_count += 1

                inputs['x'] = label_lg
                label_lg = torch.from_numpy(rand_transform(inputs)).unsqueeze(0)
                inputs['x'] = label_sg
                label_sg = torch.from_numpy(rand_transform(inputs)).unsqueeze(0)

                return data,label_lg,label_sg

        else:
            data = self.transform(src_img)
            label_object = torch.from_numpy(label_object).unsqueeze(0)
            label_background = torch.from_numpy(label_background).unsqueeze(0)

            return data,label_object,label_background
    
    def __len__(self):
        return len(self.img_list)

def get_obejct_background(img, label, kernel_size):

    img_height, img_width = img.shape[0], img.shape[1]
    radius = kernel_size//2
    label_background = np.ones([img_height,img_width],dtype = np.float32)
    label_object = np.zeros([img_height,img_width], dtype = np.float32)

    if label.size == 0:
        no_label = True
    elif len(label.shape) ==1:
        index = (min(int(label[1]+0.5),img_height-1),min(int(label[0]+0.5),img_width-1))
        r1 = max(0, index[0] - radius)
        r2 = min(index[0] + radius + 1, img_height-1)
        c1 = max(0, index[1] - radius)
        c2 = min(index[1] + radius + 1, img_width-1)
        label_background[r1:r2,c1:c2] = 0  
        label_object[index[0],index[1]] = 1
    else:
        for i in range(len(label)):
            index = (min(int(label[i][1]+0.5),img_height-1),min(int(label[i][0]+0.5),img_width-1))

            r1 = max(0, index[0] - radius)
            r2 = min(index[0] + radius + 1, img_height-1)
            c1 = max(0, index[1] - radius)
            c2 = min(index[1] + radius + 1, img_width-1)
            label_background[r1:r2,c1:c2] = 0  
            label_object[index[0],index[1]] = 1

    return label_object, label_background

def randTransform(data, device, trans_mode_default = None):
    
    batchsize = data.shape[0]
    mode_set = ['v','h','vh','r9','r27','r9h','r9v']
    #mode_set = ['v','h','vh']

    trans_mode = []
    trans_data = np.zeros(data.shape).astype(np.float32)

    for i in range(batchsize):
        if trans_mode_default != None:
            rand_transform_mode = trans_mode_default[i]
        else:
            rand_transform_mode = mode_set[np.random.permutation(len(mode_set))[0]]
        trans_mode.append(rand_transform_mode)

        x_0 = data[i,:,:,:].cpu().numpy()
        x_1 = np.zeros((x_0.shape[1],x_0.shape[2],x_0.shape[0]))

        for j in range(x_0.shape[0]):
            x_1[:,:,j] = x_0[j,:,:]

        transformer = RandomTransform()
        inputs = {}
        inputs['transform_mode'] = rand_transform_mode
        inputs['x'] = x_1
        x_trans = transformer(inputs)
        if len(x_trans.shape) == 2:
            x_trans = np.expand_dims(x_trans, axis = 2)

        for j in range(x_trans.shape[2]):
            trans_data[i,j,:,:] = x_trans[:,:,j]
    
    trans_data = torch.from_numpy(trans_data).to(device)
    
    return trans_data, trans_mode

class RandomTransform(object):
    '''Transfrom randomly the image.
    '''
    def __call__(self, inputs):

        x = inputs['x']
        transform_mode = inputs['transform_mode']

        if transform_mode == 'unchanged':
            x = x
        elif transform_mode == 'v':  # Vertical flip
            x = cv2.flip(x,0)
        elif transform_mode == 'h':  # Horizontal flip
            x = cv2.flip(x,1)
        elif transform_mode == 'vh':  # Vertical + Horizontal flip
            x = cv2.flip(x,-1)
        elif transform_mode == 'r9':  # Rotation 90°
            x = cv2.transpose(x)
            x = cv2.flip(x,0)
        elif transform_mode == 'r27':  # Rotation 270°
            x = cv2.transpose(x)
            x = cv2.flip(x,1)
        elif transform_mode == 'r9h':  # Rotation 90° + Horizontal flip
            x = cv2.transpose(x)
            x = cv2.flip(cv2.flip(x,0),0)
        elif transform_mode == 'r9v':  # Rotation 90° + Vertical flip
            x = cv2.transpose(x)
            x = cv2.flip(cv2.flip(x,0),1)
          
        return x
