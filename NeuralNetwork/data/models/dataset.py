import os
import glob
import pickle

# import pcl
# import torch
import torch.utils.data
# import torch.nn as nn
import numpy as np
import random
# from vstsim.grasping import grasp_info

def get_file_name(file_dir_):
    file_list = []
    for root, dirs, files in os.walk(file_dir_):
        # print(root)  # current path
        if root.count('/') == file_dir_.count('/'):
            for name in files:
                str = file_dir_ + '/' + name
                file_list.append(str)
    file_list.sort()
    return file_list


class DexterousVacuumGraspOneViewDataset(torch.utils.data.Dataset):
    def __init__(self, path_grasps, path_pcs, resolution=24,
                 flg_shuffle=True,
                 flg_normalize=False,
                 flg_smooth=True, thresh_max_num=50000,
                 flg_cut_value=False, thresh_min_quality=0.0, thresh_max_quality=1.0,
                 num_classes=5, ranges_clss=np.array([0.0])):
        self.path_grasps = path_grasps
        self.path_pcs = path_pcs
        self.num_classes = num_classes
        self.ranges_clss = ranges_clss

        # projection related
        self.flg_shuffle = flg_shuffle
        self.flg_normalize = flg_normalize
        self.flg_smooth = flg_smooth
        self.resolution = resolution

        self.cnt_0001 = 0
        self.cnt_0102 = 0
        self.cnt_0203 = 0
        self.cnt_0304 = 0
        self.cnt_0405 = 0
        self.cnt_0506 = 0
        self.cnt_0607 = 0
        self.cnt_0708 = 0
        self.cnt_0809 = 0
        self.cnt_0910 = 0
        self.thresh_0001 = 1 * thresh_max_num
        self.thresh_0102 = 1 * thresh_max_num
        self.thresh_0203 = 1 * thresh_max_num
        self.thresh_0304 = 1 * thresh_max_num
        self.thresh_0405 = 1 * thresh_max_num
        self.thresh_0506 = 1 * thresh_max_num
        self.thresh_0607 = 1 * thresh_max_num
        self.thresh_0708 = 1 * thresh_max_num
        self.thresh_0809 = 1 * thresh_max_num
        self.thresh_0910 = 1 * thresh_max_num

        self.grasps = []

        self.grasps_list_all = get_file_name(self.path_grasps)
        self.object_numbers = self.grasps_list_all.__len__()

        self.amount = 0

        self.max_x_ = -np.inf
        self.max_y_ = -np.inf
        self.max_z_ = -np.inf
        self.min_x_ = np.inf
        self.min_y_ = np.inf
        self.min_z_ = np.inf

        # tmp_list = []
        for i, path_grasp in enumerate(self.grasps_list_all):
            lst_grasp = pickle.load(open(path_grasp, 'rb'))
            for cnt, g_info in enumerate(lst_grasp):
                flg_pc, pc = \
                    self.get_point_cloud(self.path_pcs + '/r' + str(self.resolution) + '/' +
                                         str(g_info.get_name_grasp()) + '.npy')
                if not flg_pc:
                    continue

                if flg_cut_value:
                    if g_info.quality_grasp > thresh_max_quality or \
                            g_info.quality_grasp < thresh_min_quality:
                        continue

                if self.flg_smooth:
                    if g_info.quality_grasp <= 0.1 and self.cnt_0001 < self.thresh_0001:
                        self.cnt_0001 += 1
                    elif 0.1 < g_info.quality_grasp <= 0.2 and self.cnt_0102 < self.thresh_0102:
                        self.cnt_0102 += 1
                    elif 0.2 < g_info.quality_grasp <= 0.3 and self.cnt_0203 < self.thresh_0203:
                        self.cnt_0203 += 1
                    elif 0.3 < g_info.quality_grasp <= 0.4 and self.cnt_0304 < self.thresh_0304:
                        self.cnt_0304 += 1
                    elif 0.4 < g_info.quality_grasp <= 0.5 and self.cnt_0405 < self.thresh_0405:
                        self.cnt_0405 += 1
                    elif 0.5 < g_info.quality_grasp <= 0.6 and self.cnt_0506 < self.thresh_0506:
                        self.cnt_0506 += 1
                    elif 0.6 < g_info.quality_grasp <= 0.7 and self.cnt_0607 < self.thresh_0607:
                        self.cnt_0607 += 1
                    elif 0.7 < g_info.quality_grasp <= 0.8 and self.cnt_0708 < self.thresh_0708:
                        self.cnt_0708 += 1
                    elif 0.8 < g_info.quality_grasp <= 0.9 and self.cnt_0809 < self.thresh_0809:
                        self.cnt_0809 += 1
                    elif 0.9 < g_info.quality_grasp <= 1.0 and self.cnt_0910 < self.thresh_0910:
                        self.cnt_0910 += 1
                    else:
                        continue

                else:
                    if g_info.quality_grasp <= 0.1:
                        self.cnt_0001 += 1
                    elif g_info.quality_grasp <= 0.2:
                        self.cnt_0102 += 1
                    elif g_info.quality_grasp <= 0.3:
                        self.cnt_0203 += 1
                    elif g_info.quality_grasp <= 0.4:
                        self.cnt_0304 += 1
                    elif g_info.quality_grasp <= 0.5:
                        self.cnt_0405 += 1
                    elif g_info.quality_grasp <= 0.6:
                        self.cnt_0506 += 1
                    elif g_info.quality_grasp <= 0.7:
                        self.cnt_0607 += 1
                    elif g_info.quality_grasp <= 0.8:
                        self.cnt_0708 += 1
                    elif g_info.quality_grasp <= 0.9:
                        self.cnt_0809 += 1
                    else:
                        self.cnt_0910 += 1

                tmp_grasp = {'pc': 1.0 * pc, 'quality': 1.0 * g_info.quality_grasp}
                self.grasps.append(tmp_grasp)
                self.amount += 1

                if np.max(pc[0, :, :]) > self.max_x_: self.max_x_ = np.max(pc[0, :, :])
                if np.max(pc[1, :, :]) > self.max_y_: self.max_y_ = np.max(pc[1, :, :])
                if np.max(pc[2, :, :]) > self.max_z_: self.max_z_ = np.max(pc[2, :, :])

                if np.min(pc[0, :, :]) < self.min_x_: self.min_x_ = np.min(pc[0, :, :])
                if np.min(pc[1, :, :]) < self.min_y_: self.min_y_ = np.min(pc[1, :, :])
                if np.min(pc[2, :, :]) < self.min_z_: self.min_z_ = np.min(pc[2, :, :])

            print("loaded: ", path_grasp)

        if self.flg_shuffle:
            random.shuffle(self.grasps)

        self.matrix_min_x = self.min_x_ * np.ones([self.resolution, self.resolution], dtype=np.float32)
        self.matrix_norm_factor_x = \
            np.ones([self.resolution, self.resolution], dtype=np.float32) / (self.max_x_-self.min_x_)
        self.matrix_min_y = self.min_y_ * np.ones([self.resolution, self.resolution], dtype=np.float32)
        self.matrix_norm_factor_y = \
            np.ones([self.resolution, self.resolution], dtype=np.float32) / (self.max_y_-self.min_y_)
        self.matrix_min_z = self.min_z_ * np.ones([self.resolution, self.resolution], dtype=np.float32)
        self.matrix_norm_factor_z = \
            np.ones([self.resolution, self.resolution], dtype=np.float32) / (self.max_z_-self.min_z_)

        """
        thresholds for labels
        """
        self.thresh_class = 1.0 / float(self.num_classes)
        self.array_thresh = np.zeros(self.num_classes+1)
        if np.sum(self.ranges_clss) == 0.0:
            for i in range(1, self.num_classes+1):
                self.array_thresh[i] = self.array_thresh[i-1] + self.thresh_class
        else:
            self.array_thresh = 1.0 * self.ranges_clss

    def get_point_cloud(self, path):
        try:
            pc = np.load(path)
            return True, pc
        except:
            try:
                pc = np.load(path)
                return True, pc
            except:
                return False, np.zeros([1, 3])

    def __len__(self):
        return self.amount

    def __getitem__(self, index):
        grasp_pc = np.copy(self.grasps[index]['pc'])
        quality = 1.0 * self.grasps[index]['quality']
        if self.flg_normalize:
            grasp_pc[0, :, :] = (grasp_pc[0, :, :] - self.matrix_min_x) * self.matrix_norm_factor_x
            grasp_pc[1, :, :] = (grasp_pc[1, :, :] - self.matrix_min_y) * self.matrix_norm_factor_y
            grasp_pc[2, :, :] = (grasp_pc[2, :, :] - self.matrix_min_z) * self.matrix_norm_factor_z

        for i in range(1, self.num_classes+1):
            if quality < self.array_thresh[i]:
                label = i - 1
                break
        return grasp_pc, quality.astype(np.float32), label

''''''
if __name__ == '__main__':
    home_dir = os.environ['HOME']
    path_grasps = home_dir + '/Dexterous_grasp_01/dataset/test'
    path_pcs = home_dir + '/Dexterous_grasp_01/dataset'
    a = \
        DexterousVacuumGraspOneViewDataset(path_grasps=path_grasps,
                                           path_pcs=path_pcs,
                                           num_classes=5,
                                           flg_smooth=True,
                                           thresh_max_num=3000,
                                           flg_normalize=False,
                                           flg_cut_value=False,
                                           thresh_min_quality=0.5,
                                           thresh_max_quality=0.99)

    print("Test function __len__()", a.__len__())
    print("Test function __getitem__()", a[0])
    for i in range(0, a.__len__()):
        _, quality, _ = a[i]
        if quality > 0.99 or quality < 0.5:
            print("!~~~~~~~~~~~~~~!")

    for i in range(0, a.__len__()):
        _, quality, _ = a[i]
        if quality < 0.2:
            print("!!!!!!!!!!!!")
    # b = np.copy(a[0])
    # print(b)
    print("END.")

