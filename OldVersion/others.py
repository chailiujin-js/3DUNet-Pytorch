# 修改后的代码
# def recompone_result(self):
#     patch_s = self.result.shape[2]
#     #print("Patch_s value:", patch_s)  # 打印patch_s的值
#
#     N_patches_img = (self.padding_shape[0] - patch_s) // self.cut_stride + 1
#     assert (self.result.shape[0] == N_patches_img)
#
#     full_prob = torch.zeros((self.n_labels, self.padding_shape[0], self.ori_shape[1],
#                              self.ori_shape[2]))  # 初始化全概率张量
#     full_sum = torch.zeros((self.n_labels, self.padding_shape[0], self.ori_shape[1], self.ori_shape[2]))
#
#     #print("Full_prob shape: {}".format(full_prob.shape))  # 打印full_prob形状
#     #print("Full_sum shape: {}".format(full_sum.shape))  # 打印full_sum形状
#
#     for s in range(N_patches_img):
#         #print("Adding patch {} to full_prob".format(s))
#         #print("Patch {} shape: {}".format(s, self.result[s].shape))  # 打印当前要添加的patch形状
#
#         patch_shape = self.result[s].shape
#         target_shape = full_prob[:, s * self.cut_stride:s * self.cut_stride + patch_s].shape
#         if patch_shape[2:] != target_shape[2:]:  # 比较除通道维度外的其他维度尺寸是否一致
#             # 如果不一致，进行维度调整
#             if patch_shape[2] > target_shape[2]:
#                 self.result[s] = self.result[s][:, :, :target_shape[2], :, :]
#             elif patch_shape[2] < target_shape[2]:
#                 temp_patch = torch.zeros((self.result[s].shape[0], self.result[s].shape[1], target_shape[2],
#                                           self.result[s].shape[3], self.result[s].shape[4]))
#                 temp_patch[:, :, :patch_shape[2], :, :] = self.result[s]
#                 self.result[s] = temp_patch
#             if patch_shape[3] > target_shape[3]:
#                 self.result[s] = self.result[s][:, :, :, :target_shape[3], :]
#             elif patch_shape[3] < target_shape[3]:
#                 temp_patch = torch.zeros((self.result[s].shape[0], self.result[s].shape[1], self.result[s].shape[2],
#                                           target_shape[3], self.result[s].shape[4]))
#                 temp_patch[:, :, :, :patch_shape[3], :] = self.result[s]
#                 self.result[s] = temp_patch
#             if patch_shape[4] > target_shape[4]:
#                 self.result[s] = self.result[s][:, :, :, :, :target_shape[4]]
#             elif patch_shape[4] < target_shape[4]:
#                 temp_patch = torch.zeros((self.result[s].shape[0], self.result[s].shape[1], self.result[s].shape[2],
#                                           self.result[s].shape[3], target_shape[4]))
#                 temp_patch[:, :, :, :, :patch_shape[4]] = self.result[s]
#                 self.result[s] = temp_patch
#
#         full_prob[:, s * self.cut_stride:s * self.cut_stride + patch_s] += self.result[s]
#         full_sum[:, s * self.cut_stride:s * self.cut_stride + patch_s] += 1
#
#     assert (torch.min(full_sum) >= 1.0)  # 至少有一个
#     final_avg = full_prob / full_sum
#     print("Final_avg shape: {}".format(final_avg.shape))  # 打印final_avg形状
#     assert (torch.max(final_avg) <= 1.0)  # 像素最大值为1.0
#     assert (torch.min(final_avg) >= 0.0)  # 像素最小值为0.0
#     img = final_avg[:, :self.ori_shape[0], :self.ori_shape[1], :self.ori_shape[2]]
#     return img.unsqueeze(0)