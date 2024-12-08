from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import config
from utils import logger,common
from dataset.dataset_lits_test import Test_Datasets,to_one_hot_3d
import SimpleITK as sitk
import os
import numpy as np
# from models import ResUNet
from my_UNet import UNet3D
from utils.metrics import DiceAverage
from collections import OrderedDict

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 假设要和train.py对应使用cuda:1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def predict_one_img(model, img_dataset, args):
    dataloader = DataLoader(dataset=img_dataset, batch_size=1, num_workers=0, shuffle=False)
    model.eval()
    test_dice = DiceAverage(args.n_labels)
    target = to_one_hot_3d(img_dataset.label, args.n_labels)
    print("Target device:", target.device)  # 添加打印语句查看target的设备

    with torch.no_grad():
        for data in tqdm(dataloader,total=len(dataloader)):
            data = data.to(device)
            print("Data device:", data.device)  # 添加打印语句查看data的设备
            output = model(data)
            # output = nn.functional.interpolate(output, scale_factor=(1//args.slice_down_scale,1//args.xy_down_scale,1//args.xy_down_scale), mode='trilinear', align_corners=False) # 空间分辨率恢复到原始size
            img_dataset.update_result(output.detach().cpu())

    pred = img_dataset.recompone_result()
    pred = torch.argmax(pred,dim=1)

    pred_img = common.to_one_hot_3d(pred,args.n_labels)
    test_dice.update(pred_img, target)
    
    test_dice = OrderedDict({'Dice_liver': test_dice.avg[1]})
    if args.n_labels==3: test_dice.update({'Dice_tumor': test_dice.avg[2]})
    
    pred = np.asarray(pred.numpy(),dtype='uint8')
    if args.postprocess:
        pass # TO DO
    pred = sitk.GetImageFromArray(np.squeeze(pred,axis=0))

    return test_dice, pred

if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./experiments', args.save)
    device = torch.device('cpu' if args.cpu else 'cuda:0')
    # model info
    model = UNet3D(in_channels=1, out_channels=args.n_labels).to(device)
    print(args.gpu_id)
    #model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU
    # ckpt = torch.load('{}/best_model.pth'.format(save_path))
    # model.load_state_dict(ckpt['net'])
    ckpt = torch.load('{}/best_model.pth'.format(save_path))
    state_dict = ckpt['net']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)


    test_log = logger.Test_Logger(save_path,"test_log")
    # data info
    result_save_path = '{}/result'.format(save_path)
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    print("测试数据路径: ", args.test_data_path)
    datasets = Test_Datasets(args.test_data_path,args=args)
    for img_dataset,file_idx in datasets:
        test_dice,pred_img = predict_one_img(model, img_dataset, args)
        test_log.update(file_idx, test_dice)
        sitk.WriteImage(pred_img, os.path.join(result_save_path, 'result-'+file_idx+'.gz'))
