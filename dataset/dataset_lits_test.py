from torch._C import dtype

import config
from utils.common import *
from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch, os
from torch.utils.data import Dataset, DataLoader
from glob import glob
import math
import SimpleITK as sitk

# os.chdir('/home/dcd/gy/3DUnet')
#print("当前工作目录:", os.getcwd())
class Img_DataSet(Dataset):
    def __init__(self, data_path, label_path, args):
        self.n_labels = args.n_labels
        self.cut_size = args.test_cut_size
        self.cut_stride = args.test_cut_stride

        # 读取一个data文件并归一化 、resize
        self.ct = sitk.ReadImage(data_path,sitk.sitkInt16)
        self.data_np = sitk.GetArrayFromImage(self.ct)
        self.ori_shape = self.data_np.shape
        self.data_np = ndimage.zoom(self.data_np, (args.slice_down_scale, args.xy_down_scale, args.xy_down_scale), order=3) # 双三次重采样
        self.data_np[self.data_np > args.upper] = args.upper
        self.data_np[self.data_np < args.lower] = args.lower
        self.data_np = self.data_np/args.norm_factor
        self.resized_shape = self.data_np.shape
        # 扩展一定数量的slices，以保证卷积下采样合理运算
        self.data_np = self.padding_img(self.data_np, self.cut_size,self.cut_stride)
        self.padding_shape = self.data_np.shape
        # 对数据按步长进行分patch操作，以防止显存溢出
        self.data_np = self.extract_ordered_overlap(self.data_np, self.cut_size, self.cut_stride)

        # 读取一个label文件 shape:[s,h,w]
        self.seg = sitk.ReadImage(label_path,sitk.sitkInt8)
        self.label_np = sitk.GetArrayFromImage(self.seg)
        if self.n_labels==2:
            self.label_np[self.label_np > 0] = 1
        self.label = torch.from_numpy(np.expand_dims(self.label_np,axis=0)).long()

        # 预测结果保存
        self.result = None

    def __getitem__(self, index):
        data = torch.from_numpy(self.data_np[index])
        data = torch.FloatTensor(data).unsqueeze(0)
        return data

    def __len__(self):
        return len(self.data_np)

    def update_result(self, tensor):
        #以下两行设备代码于2024.12.9添加
        if tensor.device != torch.device('cuda'):
            tensor = tensor.to('cuda')
        # tensor = tensor.detach().cpu() # shape: [N,class,s,h,w]
        # tensor_np = np.squeeze(tensor_np,axis=0)
        if self.result is not None:
            self.result = torch.cat((self.result, tensor), dim=0)
        else:
            self.result = tensor

    # 原始代码
    def recompone_result(self):

        # 以下四行设备代码于2024.12.9添加
        if self.result is None:
            return None  # 若self.result为空，直接返回None，避免后续报错
        if self.result.device != torch.device('cuda'):
            self.result = self.result.to('cuda')

        patch_s = self.result.shape[2]

        N_patches_img = (self.padding_shape[0] - patch_s) // self.cut_stride + 1
        assert (self.result.shape[0] == N_patches_img)

        full_prob = torch.zeros((self.n_labels, self.padding_shape[0], self.ori_shape[1],self.ori_shape[2]))  # itialize to zero mega array with sum of Probabilities
        print('full_prob：', full_prob.shape)

        full_sum = torch.zeros((self.n_labels, self.padding_shape[0], self.ori_shape[1], self.ori_shape[2]))
        print('full_sum：', full_sum.shape)
        for s in range(N_patches_img):
            print('############', self.result[s].shape, full_prob[:, s * self.cut_stride:s * self.cut_stride + patch_s].shape)
            if full_prob[:, s * self.cut_stride:s * self.cut_stride + patch_s].device != self.result[s].device:
                print("Error: full_prob and self.result[s] are on different devices!")
                print("full_prob device:", full_prob[:, s * self.cut_stride:s * self.cut_stride + patch_s].device)
                print("self.result[s] device:", self.result[s].device)
            full_prob[:, s * self.cut_stride:s * self.cut_stride + patch_s] += self.result[s].to("cuda:0")
            full_sum[:, s * self.cut_stride:s * self.cut_stride + patch_s] += 1

        assert (torch.min(full_sum) >= 1.0)  # at least one
        final_avg = full_prob / full_sum
        # print(final_avg.size())
        assert (torch.max(final_avg) <= 1.0)  # max value for a pixel is 1.0
        assert (torch.min(final_avg) >= 0.0)  # min value for a pixel is 0.0
        img = final_avg[:, :self.ori_shape[0], :self.ori_shape[1], :self.ori_shape[2]]
        return img.unsqueeze(0)

    def padding_img(self, img, size, stride):
        assert (len(img.shape) == 3)  # 3D array
        img_s, img_h, img_w = img.shape
        leftover_s = (img_s - size) % stride

        if (leftover_s != 0):
            s = img_s + (stride - leftover_s)
        else:
            s = img_s

        tmp_full_imgs = np.zeros((s, img_h, img_w),dtype=np.float32)
        tmp_full_imgs[:img_s] = img
        print("Padded images shape: " + str(tmp_full_imgs.shape))
        return tmp_full_imgs
    
    # Divide all the full_imgs in pacthes
    def extract_ordered_overlap(self, img, size, stride):
        img_s, img_h, img_w = img.shape
        assert (img_s - size) % stride == 0
        N_patches_img = (img_s - size) // stride + 1

        print("Patches number of the image:{}".format(N_patches_img))
        patches = np.empty((N_patches_img, size, img_h, img_w), dtype=np.float32)

        for s in range(N_patches_img):  # loop over the full images
            patch = img[s * stride : s * stride + size]
            #print("Patch {} shape: {}".format(s, patch.shape))  # 打印每个patch的形状
            patches[s] = patch

        return patches  # array with all the full_imgs divided in patches

def Test_Datasets(dataset_path, args):
    data_list = sorted(glob(os.path.join(dataset_path, 'ct/*')))
    label_list = sorted(glob(os.path.join(dataset_path, 'label/*')))
    print("The number of test samples is: ", len(data_list))
    for datapath, labelpath in zip(data_list, label_list):
        print("\nStart Evaluate: ", datapath)
        yield Img_DataSet(datapath, labelpath,args=args), datapath.split('-')[-1]


# if __name__ == "__main__":
#     import argparse
#     # 创建参数解析器
#     parser = argparse.ArgumentParser(description='Test dataset loading')
#     # 添加必要的参数，这里假设你的 `Img_DataSet` 类初始化需要的参数，你可以根据实际情况进行调整
#     parser.add_argument('--test_cut_size', type=int, default=64, help='Cut size for testing')
#     parser.add_argument('--test_cut_stride', type=int, default=32, help='Cut stride for testing')
#     parser.add_argument('--slice_down_scale', type=float, default=0.5, help='Slice down scale factor')
#     parser.add_argument('--xy_down_scale', type=float, default=0.5, help='XY down scale factor')
#     parser.add_argument('--upper', type=float, default=200, help='Upper bound for normalization')
#     parser.add_argument('--lower', type=float, default=-200, help='Lower bound for normalization')
#     parser.add_argument('--norm_factor', type=float, default=400, help='Normalization factor')
#     parser.add_argument('--n_labels', type=int, default=2, help='Number of labels')
#     parser.add_argument('--test_data_path', type=str, default='./raw_dataset/test', help='Path to the test dataset')
#     args = parser.parse_args()
#
#     test_datasets = Test_Datasets(dataset_path=args.test_data_path, args=args)
#     for img_dataset, file_idx in test_datasets:
#         print("当前样本索引:", file_idx)
#         print("数据集长度:", len(img_dataset))
#         sample_data = img_dataset[0]  # 获取第一个数据样本
#         print("第一个数据样本形状:", sample_data.shape)
#         # 可以在这里添加更多对数据集相关属性、方法的测试操作，比如查看标签形状等
#         label = img_dataset.label
#         print("标签形状:", label.shape)
#
#         try:
#             recomposed_result = img_dataset.recompone_result()
#             print("重组结果形状:", recomposed_result.shape)
#         except Exception as e:
#             print("调用recompone_result方法出现错误:", e)

if __name__ == "__main__":
    args = config.args
    dataset_path = args.test_data_path  # 替换为实际的数据集路径
    dataset_generator = Test_Datasets(dataset_path, args)
    for dataset, file_idx in dataset_generator:
        result = dataset.recompone_result()
        if result is not None:
            print(f"Result shape for file {file_idx}: {result.shape}")
            # 这里可以根据需要进一步对结果进行处理，比如保存结果等
            # 例如，如果要保存为Numpy数组，可以使用：
            # np.save(f"result_{file_idx}.npy", result.squeeze(0).numpy())
            # 如果要转换为SimpleITK图像对象保存（假设符合相应的图像格式要求等）：
            # result_sitk = sitk.GetImageFromArray(result.squeeze(0).numpy())
            # sitk.WriteImage(result_sitk, f"result_{file_idx}.nii.gz")
        else:
            print(f"Result is None for file {file_idx}")