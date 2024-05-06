import os, random
import numpy as np
import pandas as pd
import torch

from lib.poses import utils
from lib.datasets import image_utils
from lib.datasets.linemod import inout
from lib.datasets.linemod.dataloader_query import LINEMOD

np.random.seed(2024)
random.seed(2024)
number_train_template = 1542
number_test_template = 301


class TemplatesLINEMOD(LINEMOD):
    def __init__(self, root_dir, dataset, list_id_obj, split, image_size, mask_size, im_transform, save_path):
        self.root_dir = root_dir
        self.dataset_name = dataset
        self.list_id_obj = list(list_id_obj)
        self.split = split
        assert self.split == "test", print("Split should be test")
        self.image_size = image_size
        self.mask_size = mask_size
        self.save_path = save_path
        self.query_data, self.template_data = self.get_data()
        self.im_transform = im_transform
        print("Length of the dataset: {}".format(self.__len__()))
        self.save_random_sequences()

    def __len__(self):
        return len(self.query_data)

    def get_data(self):
        list_path, list_poses, ids_obj, id_symmetry = [], [], [], []
        if os.path.exists("./lib/poses/predefined_poses/half_sphere_level2.npy"):
            obj_poses = np.load("./lib/poses/predefined_poses/half_sphere_level2.npy")
        else:
            obj_poses = np.load("../lib/poses/predefined_poses/half_sphere_level3.npy")
        obj_locations = []
        for id_frame in range(number_test_template):
            location = utils.opencv2opengl(np.asarray(obj_poses[id_frame]))[2, :3]
            obj_locations.append(torch.from_numpy(np.round(location, 4).reshape(3)))

        for id_obj in self.list_id_obj:
            obj_name = inout.LINEMOD_real_id_to_name[id_obj]
            for id_frame in range(number_test_template):
                list_path.append(os.path.join("templatesLINEMOD/test", obj_name, "{:06d}.png".format(id_frame)))
                list_poses.append(obj_locations[id_frame].reshape(-1, 3))
                ids_obj.append(id_obj)
                id_symmetry.append(inout.list_all_id_symmetry[id_obj])
            all_data = {"id_obj": ids_obj,
                        "id_symmetry": id_symmetry,
                        "obj_poses": list_poses,
                        "synthetic_path": list_path}
        template_frame = pd.DataFrame.from_dict(all_data, orient='index')
        template_frame = template_frame.transpose()
        return template_frame, template_frame

    def _sample_template(self, idx, save_path=None):
        img = self._sample(idx, isQuery=False, isPositive=True)
        mask = self._sample_mask(idx)
        full_res_mask = mask
        full_res_mask = (np.asarray(full_res_mask) / 255. > 0) * 1
        full_res_mask = torch.from_numpy(full_res_mask).unsqueeze(0)
        if save_path is None:
            mask = image_utils.process_mask_image(mask, mask_size=self.mask_size)
            return [self.im_transform(img), mask, full_res_mask]
        else:
            img.save(save_path + ".png")
            mask.save(save_path + "_mask.png")

    def __getitem__(self, idx):
        id_obj = self.query_data.iloc[idx]['id_obj']
        id_symmetry = inout.list_all_id_symmetry[id_obj]
        obj_pose = self.query_data.iloc[idx]["obj_poses"].reshape(3)
        template, mask, full_res_mask = self._sample_template(idx)
        return dict(id_obj=id_obj, id_symmetry=id_symmetry, obj_pose=obj_pose,
                    template=template, mask=mask, full_res_mask=full_res_mask)

    def save_random_sequences(self):
        len_data = self.__len__()
        list_index = np.unique(np.random.randint(0, len_data, 10))
        print("Saving samples at {}".format(self.save_path))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        for idx in list_index:
            save_path = os.path.join(self.save_path, "{:06d}".format(idx))
            self._sample_template(idx, save_path)
