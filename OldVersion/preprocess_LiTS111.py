# 导入必要的库
import numpy as np  # 用于数组和数学运算
import os  # 用于处理文件和目录
import SimpleITK as sitk  # 用于医学图像处理
import random  # 用于随机数生成
from scipy import ndimage  # 用于图像处理
from os.path import join  # 用于路径连接
import config  # 导入配置文件

# 打印当前工作目录
print("当前工作目录：", os.getcwd())


# 定义数据预处理类
class LITS_preprocess:
    def __init__(self, raw_dataset_path, fixed_dataset_path, args):
        # 初始化类的属性
        self.raw_root_path = raw_dataset_path  # 原始数据集路径
        self.fixed_path = fixed_dataset_path  # 处理后数据集路径
        self.classes = args.n_labels  # 分割类别数（2或3）
        self.upper = args.upper  # 灰度值上限
        self.lower = args.lower  # 灰度值下限
        self.expand_slice = args.expand_slice  # 轴向外侧扩张的slice数量
        self.size = args.min_slices  # 取样的slice数量
        self.xy_down_scale = args.xy_down_scale  # x和y轴的降采样比例
        self.slice_down_scale = args.slice_down_scale  # slice轴的降采样比例
        self.valid_rate = args.valid_rate  # 验证集比例

    def fix_data(self):
        # 创建保存目录，如果不存在
        if not os.path.exists(self.fixed_path):
            os.makedirs(join(self.fixed_path, 'ct'))  # 创建CT图像目录
            os.makedirs(join(self.fixed_path, 'label'))  # 创建标签目录

        # 列出原始CT图像文件
        file_list = os.listdir(join(self.raw_root_path, 'ct'))
        Numbers = len(file_list)  # 计算样本总数
        print('Total numbers of samples is :', Numbers)  # 打印样本总数

        # 遍历每个CT文件
        for ct_file, i in zip(file_list, range(Numbers)):
            print("==== {} | {}/{} ====".format(ct_file, i + 1, Numbers))  # 打印当前处理的文件信息
            ct_path = os.path.join(self.raw_root_path, 'ct', ct_file)  # CT文件路径
            seg_path = os.path.join(self.raw_root_path, 'label', ct_file.replace('volume', 'segmentation'))  # 标签路径

            # 处理CT图像和标签
            new_ct, new_seg = self.process(ct_path, seg_path, classes=self.classes)

            # 如果处理成功，则保存新图像
            if new_ct is not None and new_seg is not None:
                sitk.WriteImage(new_ct, os.path.join(self.fixed_path, 'ct', ct_file))  # 保存CT图像
                sitk.WriteImage(new_seg, os.path.join(self.fixed_path, 'label',
                                                      ct_file.replace('volume', 'segmentation').replace('.nii',
                                                                                                        '.nii.gz')))  # 保存标签图像

    def process(self, ct_path, seg_path, classes=None):
        # 读取CT图像和标签
        ct = sitk.ReadImage(ct_path, sitk.sitkInt16)  # 读取CT图像
        ct_array = sitk.GetArrayFromImage(ct)  # 转换为数组
        seg = sitk.ReadImage(seg_path, sitk.sitkInt8)  # 读取标签图像
        seg_array = sitk.GetArrayFromImage(seg)  # 转换为数组

        print("Ori shape:", ct_array.shape, seg_array.shape)  # 打印原始形状
        if classes == 2:
            # 将肝脏和肝肿瘤的标签融合为一个
            seg_array[seg_array > 0] = 1  # 将所有非零标签设置为1

        # 截断灰度值在指定阈值之外
        ct_array[ct_array > self.upper] = self.upper  # 上限截断
        ct_array[ct_array < self.lower] = self.lower  # 下限截断

        # 对x和y轴进行降采样，slice轴的spacing归一化到slice_down_scale
        ct_array = ndimage.zoom(ct_array,
                                (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale),
                                order=3)
        seg_array = ndimage.zoom(seg_array,
                                 (ct.GetSpacing()[-1] / self.slice_down_scale, self.xy_down_scale, self.xy_down_scale),
                                 order=0)

        # 找到肝脏区域开始和结束的slice，并各向外扩张
        z = np.any(seg_array, axis=(1, 2))  # 检查每个slice是否有肝脏区域
        start_slice, end_slice = np.where(z)[0][[0, -1]]  # 找到开始和结束的slice索引

        # 两个方向上各扩张个slice
        if start_slice - self.expand_slice < 0:
            start_slice = 0  # 防止下界越界
        else:
            start_slice -= self.expand_slice  # 向外扩张

        if end_slice + self.expand_slice >= seg_array.shape[0]:
            end_slice = seg_array.shape[0] - 1  # 防止上界越界
        else:
            end_slice += self.expand_slice  # 向外扩张

        print("Cut out range:", str(start_slice) + '--' + str(end_slice))  # 打印切割范围

        # 如果剩余的slice数量不足size，直接放弃
        if end_slice - start_slice + 1 < self.size:
            # print('Too little slice，give up the sample:', ct_file)  # 打印放弃的信息
            return None, None  # 返回None表示放弃该样本

        # 截取保留区域
        ct_array = ct_array[start_slice:end_slice + 1, :, :]  # 截取CT图像
        seg_array = seg_array[start_slice:end_slice + 1, :, :]  # 截取标签图像
        print("Preprocessed shape:", ct_array.shape, seg_array.shape)  # 打印预处理后的形状

        # 保存为对应的格式
        new_ct = sitk.GetImageFromArray(ct_array)  # 将数组转换为图像
        new_ct.SetDirection(ct.GetDirection())  # 设置方向
        new_ct.SetOrigin(ct.GetOrigin())  # 设置原点
        new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale),
                           ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))  # 设置spacing

        new_seg = sitk.GetImageFromArray(seg_array)  # 将标签数组转换为图像
        new_seg.SetDirection(ct.GetDirection())  # 设置方向
        new_seg.SetOrigin(ct.GetOrigin())  # 设置原点
        new_seg.SetSpacing((ct.GetSpacing()[0] * int(1 / self.xy_down_scale),
                            ct.GetSpacing()[1] * int(1 / self.xy_down_scale), self.slice_down_scale))  # 设置spacing

        return new_ct, new_seg  # 返回处理后的CT图像和标签

    def write_train_val_name_list(self):
        # 列出固定数据集中CT图像的名称
        data_name_list = os.listdir(join(self.fixed_path, "ct"))
        data_num = len(data_name_list)  # 计算样本数量
        print('the fixed dataset total numbers of samples is :', data_num)  # 打印样本数量

        random.shuffle(data_name_list)  # 随机打乱样本顺序

        assert self.valid_rate < 1.0  # 确保验证集比例有效
        # 划分训练集和验证集
        train_name_list = data_name_list[0:int(data_num * (1 - self.valid_rate))]  # 训练集样本
        val_name_list = data_name_list[int(data_num * (1 - self.valid_rate)):int(
            data_num * ((1 - self.valid_rate) + self.valid_rate))]  # 验证集样本

        # 写入训练集和验证集的名称列表
        self.write_name_list(train_name_list, "train_path_list.txt")  # 写入训练集路径
        self.write_name_list(val_name_list, "val_path_list.txt")  # 写入验证集路径

    def write_name_list(self, name_list, file_name):
        # 将名称列表写入文件
        f = open(join(self.fixed_path, file_name), 'w')  # 打开文件以写入
        for name in name_list:
            ct_path = os.path.join(self.fixed_path, 'ct', name)  # CT图像路径
            seg_path = os.path.join(self.fixed_path, 'label', name.replace('volume', 'segmentation'))  # 标签路径
            f.write(ct_path + ' ' + seg_path + "\n")  # 写入路径
        f.close()  # 关闭文件


# 主程序入口
if __name__ == '__main__':
    raw_dataset_path = './raw_dataset/train/'  # 原始数据集路径
    fixed_dataset_path = './fixed_data/'  # 处理后数据集路径

    # raw_dataset_path = '/ssd/lzq/dataset/LiTS/train'  # 可替换的路径
    # fixed_dataset_path = '/ssd/lzq/dataset/fixed_lits'  # 可替换的路径

    args = config.args  # 从配置文件获取参数
    tool = LITS_preprocess(raw_dataset_path, fixed_dataset_path, args)  # 创建预处理工具实例
    tool.fix_data()  # 对原始图像进行修剪并保存
    tool.write_train_val_name_list()  # 创建索引txt文件
