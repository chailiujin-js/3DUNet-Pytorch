import unittest

import SimpleITK as sitk
import matplotlib.pyplot as plt


class TestNiiShow(unittest.TestCase):
    def test_show_3dimg_slice(self):
        # 读取nii格式文件
        image_path = '/home/dcd/gy/3DUnet/fixed_data/ct/volume-0.nii'
        img = sitk.ReadImage(image_path)
        # 转换为numpy数组
        image_array = sitk.GetArrayFromImage(img)
        # 显示图像
        plt.imshow(image_array[100, :, :], cmap='gray')  # 选择第100个切片
        plt.savefig("./tmp/3dimg_slice_100.png")

    def test_show_3dmask_slice(self):
        # 读取nii格式文件
        image_path = '/home/dcd/gy/3DUnet/fixed_data/label/segmentation-0.nii.gz'
        img = sitk.ReadImage(image_path)
        # 转换为numpy数组
        image_array = sitk.GetArrayFromImage(img)
        # 显示图像
        plt.imshow(image_array[100, :, :], cmap='gray')
        plt.savefig(f"./tmp/3dmask_slice_{100}.png")
        # for i in range(image_array.shape[0]):
        #     plt.imshow(image_array[100, :, :], cmap='gray')  # 选择第i个切片
        #     plt.savefig(f"./tmp/3dmask_slice_{i}.png")
        #     plt.close()
