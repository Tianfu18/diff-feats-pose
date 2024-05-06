import os
from torch.utils.data import DataLoader


def init_dataloader(dict_dataloader, batch_size, num_workers):
    names = dict_dataloader.keys()
    print("Names dataloader {}".format(names))
    results = {}
    for name in names:
        dataloader = DataLoader(dict_dataloader[name],
                                batch_size=batch_size if 'templates' in name or name == "train" else 1,
                                num_workers=num_workers, drop_last=False,
                                pin_memory=False)
        results[name] = dataloader
    return results


def write_txt(path, list_files):
    with open(path, "w") as f:
        for idx in list_files:
            f.write(idx + "\n")
        f.close()


def get_list_background_img_from_dir(background_dir):
    if not os.path.exists(os.path.join(background_dir, "list_img.txt")):
        jpgs = [os.path.join(root, file) for root, dirs, files in os.walk(background_dir)
                for file in files if file.endswith('.jpg')]
        write_txt(os.path.join(background_dir, "list_img.txt"), jpgs)
    else:
        with open(os.path.join(background_dir, "list_img.txt"), 'r') as f:
            jpgs = [x.strip() for x in f.readlines()]
    return jpgs


def sampling_k_samples(group, k=109):
    if len(group) < k:
        return group
    return group.sample(k, random_state=2024)