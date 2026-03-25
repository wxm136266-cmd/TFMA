import torch
from torch.utils.data import Dataset, DataLoader

class FewshotDataset(Dataset):
    def __init__(self, train_data, train_label, episode_num=1000, way_num=7, shot_num=1, query_num=1):
        self.train_data = train_data
        self.train_label = train_label
        self.episode_num = episode_num
        self.way_num = way_num
        self.shot_num = shot_num
        self.query_num = query_num

    def __len__(self):
        return self.episode_num

    def __getitem__(self, index):
        query_images = []
        query_targets = []
        support_images = []
        support_targets = []
        # print(self.train_label)
        label_indices = torch.randperm(len(self.train_label)).cuda()
        # print(label_indices)
        train_label_gpu = self.train_label.cuda()
        train_data_gpu = self.train_data.cuda()
        # print('train_data_gpu:',train_data_gpu.shape)
        for label_num in range(self.way_num):
            support_idxs = torch.nonzero(train_label_gpu[label_indices] == label_num, as_tuple=False).flatten()

            support_idxs = support_idxs[:self.shot_num]
            # print('support_idxs:', support_idxs)
            support_data = train_data_gpu[label_indices][support_idxs]#先通过打乱的索引对其进行重排序再提取
            # print('support_data:',support_data)
            query_idxs = torch.nonzero(train_label_gpu[label_indices] == label_num, as_tuple=False).flatten()
            query_idxs = query_idxs[~torch.isin(query_idxs, support_idxs)][:self.query_num]#移除已经被选择的样本，从未被选择的样本中进行挑选
            query_data = train_data_gpu[label_indices][query_idxs]
            query_data_targets = train_label_gpu[label_indices][query_idxs]

            query_images.append(query_data)
            query_targets.append(query_data_targets)
            support_images.append(support_data)
            support_targets.append(torch.full((self.shot_num,), label_num).cuda())

        query_images = torch.cat(query_images, dim=0)
        query_targets = torch.cat(query_targets, dim=0)
        support_images = torch.cat(support_images, dim=0)
        support_targets = torch.cat(support_targets, dim=0)

        return query_images, query_targets, support_images, support_targets


class FewshotDatasetSingleClass(Dataset):
    def __init__(self, train_data, train_label, episode_num=20, way_num=1, shot_num=5, query_num=15):
        self.train_data = train_data
        self.train_label = train_label
        self.episode_num = episode_num  # 这里是 20，表示有 20 个样本
        self.way_num = way_num  # 只有一个类
        self.shot_num = shot_num  # 支持集的大小
        self.query_num = query_num  # 查询集的大小

    def __len__(self):
        return self.episode_num  # 这里是 20，表示 20 个 episode

    def __getitem__(self, index):
        query_images = []
        query_targets = []
        support_images = []
        support_targets = []

        # 将张量移到 GPU（如果它们还没有在 GPU 上）
        train_data_gpu = self.train_data.cuda()
        train_label_gpu = self.train_label.cuda()

        # 随机打乱标签的索引
        label_indices = torch.randperm(len(train_label_gpu)).cuda()  # 确保索引也在 GPU 上

        # 选择支持集和查询集样本
        support_data = train_data_gpu[label_indices[:self.shot_num]]  # 选择支持集
        support_target = torch.full((self.shot_num,), 0).cuda()  # 所有支持集样本的标签都是 0

        query_data = train_data_gpu[label_indices[self.shot_num:self.shot_num + self.query_num]]  # 选择查询集
        query_target = torch.full((self.query_num,), 0).cuda()# 所有查询集样本的标签也是 0

        # 将数据添加到列表中
        query_images.append(query_data)
        query_targets.append(query_target)
        support_images.append(support_data)
        support_targets.append(support_target)

        # 将列表中的数据拼接成张量
        query_images = torch.cat(query_images, dim=0)
        query_targets = torch.cat(query_targets, dim=0)
        support_images = torch.cat(support_images, dim=0)
        support_targets = torch.cat(support_targets, dim=0)

        return query_images, query_targets, support_images, support_targets.cuda()



'''
  way_num = num_classes, shot_num = number samples per class

'''