import torch as t
import torchvision.transforms as transforms
import numpy as np
import random
import time
import os
import network
from loss import fullLossFunction, weakLossFunction
from util import validate, Logger
from load_data import imgDataset, randTransform

# -----------------------
rand_seed = 64678  
if rand_seed is not None:
    os.environ['PYTHONHASHSEED'] = str(rand_seed)
    np.random.seed(rand_seed)
    t.manual_seed(rand_seed)
    t.cuda.manual_seed(rand_seed)
    t.cuda.manual_seed_all(rand_seed)
    random.seed(rand_seed)
    t.backends.cudnn.deterministic = True

# set environment
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'  ## we use 2*Nvidia-3090
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
device_ids = [0,1]

if __name__ == '__main__':

    train_net = 'LIRNet'
    kernel_size = (11,11)  ## The size of square kernel for discrete object areas
    r_thd = 6  ## The size of Golden standard region for evaluation

    # Weakly supervised approach
    ws_flag = False  ## Weakly supervised flag
    if ws_flag:
        patch_size = (100,20)  ## The size of large and small patches
        ws_params = {}
        ws_params['small_label_num'] = 100
        ws_params['large_size'] = patch_size[0]
        ws_params['small_size'] = patch_size[1]

    # Other parameters
    learning_rate = 1e-4  ## Initial learning rate
    bs_train = 6  ## Batch size for training set
    bs_val = 6  ## Batch size for validation set
    epochs = 200 ## The number of whole epochs
    delay = 20  ## The number of epoches without evaluating performance in the early training period
        
    # data path
    img_dir = '../data/CA_cell/'
    train_img_dir = img_dir + 'train/imgs/'
    train_label_dir = img_dir + 'train/labels/'
    val_img_dir = img_dir + 'validate/imgs/'
    val_label_dir = img_dir + 'validate/labels/'
    test_img_dir = img_dir + 'test/imgs/'
    test_label_dir = img_dir + 'test/labels/'

    # model initialization
    if train_net == 'LIRNet':
        count_net = network.LIRNet()
        network.weights_normal_init(count_net)

    count_net = t.nn.DataParallel(count_net,device_ids = device_ids)
    count_net.to(device)
    count_net.train() 
    print(count_net)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    # load data
    trainset = imgDataset(train_img_dir, train_label_dir, patch_size, transform = transform, train=True, weakly_supervised = ws_flag, weak_param = ws_params, random_seed= rand_seed)
    valset = imgDataset(val_img_dir, val_label_dir, patch_size, transform = transform, train=False)
    testset = imgDataset(test_img_dir, test_label_dir, patch_size, transform = transform, train=False)

    trainloader = t.utils.data.DataLoader(trainset,batch_size = bs_train,pin_memory = False,
                                        shuffle = True,num_workers = 2)
    valloader = t.utils.data.DataLoader(valset, batch_size = bs_val,pin_memory = False,
                                        shuffle = True,num_workers = 2)

    testloader = t.utils.data.DataLoader(testset,batch_size = bs_val,pin_memory = False,
                                        shuffle = True,num_workers = 2)

    print('Numbers of samples in training dataset is: %d'%(len(trainset)))
    print('Numbers of samples in validation dataset is: %d'%(len(valset)))
    print('Numbers of samples in test dataset is: %d'%(len(testset)))

    # loss function initialization
    if not ws_flag:
        loss_func = fullLossFunction()
    else:
        loss_func = weakLossFunction()

    # optimization initialization
    optimizer = t.optim.Adam(count_net.parameters(), lr=learning_rate)
    
    # learning rate scheduler
    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                    mode='max',factor=0.1,patience=20,verbose=True,)

    # training log
    logger = Logger(os.path.join('result', 'log.txt'), title='cell_pulearning')
    logger.set_names(['Train Loss', 'Valid Loss', 'Valid Count','Valid MAE', 'Valid MSE','Valid F1'])
    
    # train section
    max_F1 = 0
    for epoch in range(epochs):

        print('\nEpoch: [%d | %d] LR: %.9f'%(epoch+1, epochs, optimizer.param_groups[0]['lr']))
        running_loss = 0.0
        count_net.train() 

        iterations = 0
        t_start = time.time()

        for i,data in enumerate(trainloader):
            
            '''for weakly supervised version: dot_labels means large labels, mask_labels means small labels'''
            inputs_ori,dot_labels_ori,mask_labels_ori = data[0].to(device),data[1].to(device),data[2].to(device)

            inputs_trans, transform_mode = randTransform(inputs_ori, device)
            dot_labels_trans, _ = randTransform(dot_labels_ori, device, transform_mode)
            mask_labels_trans, _ = randTransform(mask_labels_ori, device, transform_mode)

            ## merge the original image and its augmented form
            inputs = t.cat((inputs_ori, inputs_trans))
            dot_labels = t.cat((dot_labels_ori, dot_labels_trans))
            mask_labels = t.cat((mask_labels_ori, mask_labels_trans))

            optimizer.zero_grad()
            outputs_density = count_net(inputs)

            if not ws_flag:
                pair_size = kernel_size
            else:
                pair_size = patch_size
            loss = loss_func(outputs_density,dot_labels.float(),mask_labels.float(),pair_size)
            
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            iterations = i

            # show something
            print('<%d | %d>,loss: %.5f, max_value: %.5f'%(i+1,len(trainset)//bs_train+1,loss, outputs_density.max().cpu().detach().numpy()))

        t_end = time.time()
        train_loss = running_loss/(iterations+1)
        print('\n')
        print('loss: %.5f | Time per epoch: %.2f'%(train_loss,t_end-t_start))
        print('\n')
        if epoch < delay:
            continue
        val_loss,val_ct_mean, val_mae,val_mse, val_loc_eval = validate(valloader,count_net,device,epoch, kernel_size, len(valset),bs_val, r_thd)

        val_F1 = val_loc_eval['F1']

        scheduler.step(val_F1)

        logger.append([train_loss, val_loss, val_ct_mean, val_mae, val_mse, val_F1])
        print('\n')
        print('val_loss: %.5f, val_count: %.2f, val_mae: %.2f, val_mse: %.2f, val_F1: %.4f'%(val_loss,val_ct_mean, val_mae,val_mse,val_F1))

        ## save the best model so far
        is_best = val_F1 > max_F1
        max_F1 = max(val_F1, max_F1)
        if is_best:
            save_name = 'best_model.pt'
            print(save_name)
            t.save(count_net.state_dict(),save_name)

    print('Finishing Training!')

    # best model evaluation
    count_net.load_state_dict(t.load('./best_model.pt')) #checkpoint
    count_net.to(device)
    val_loss, val_ct_mean, val_mae, val_mse, val_loc_eval= validate(valloader,count_net,device,epoch, kernel_size, len(valset),bs_val, r_thd)
    test_loss, test_ct_mean, test_mae, test_mse, test_loc_eval= validate(testloader, count_net, device, epoch, kernel_size, len(testset),1, r_thd)
    
    print('\n')
    print('val_loss: %.5f, val_count: %.2f, val_mae: %.2f, val_mse: %.2f, val_F1: %.4f'%(val_loss,val_ct_mean, val_mae,val_mse, val_loc_eval['F1']))
    print('\nval_loc_eval:', val_loc_eval)
    print('\ntest_count: %.2f, test_mae: %.2f, test_mse: %.2f, test_F1: %.4f'%(test_ct_mean,test_mae,test_mse, test_loc_eval['F1']))
    print('\ntest_loc_eval:', test_loc_eval)