import argparse
import os
import numpy as np
import time
from tqdm import tqdm
import torch
# multiprocessing to accelerate the rendering
from functools import partial
import multiprocessing

from lib.utils import weights, logger
from lib.utils.config import Config
from lib.datasets.dataloader_utils import init_dataloader

from lib.models.diffusion.diffusion_extractor import DiffusionExtractor
from lib.models.diffusion.aggregation_network import AggregationNetwork
from lib.models.diffusion.stable_diffusion.resnet import collect_dims
from lib.models.diffusion_network import DiffusionFeatureExtractor

from lib.datasets.tless.dataloader_query import Tless
from lib.datasets.tless.dataloader_template import TemplatesTless
from lib.datasets.tless import testing_utils
from lib.datasets.tless.eval_utils import eval_vsd
from lib.datasets.im_transform import im_transform

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str)
parser.add_argument('--checkpoint', type=str)
args = parser.parse_args()

config_global = Config(config_file="./config.json").get_config()
config_run = Config(args.config_path).get_config()

# initialize global config for the training
dir_name = (args.config_path.split('/')[-1]).split('.')[0]
print("config", dir_name)
save_path = os.path.join(config_global.root_path, config_run.log.weights, dir_name)
trainer_dir = os.path.join(os.getcwd(), "logs")
trainer_logger = logger.init_logger(save_path=save_path,
                                    trainer_dir=trainer_dir,
                                    trainer_logger_name=dir_name)

# initialize network
torch.cuda.memory_allocated()
diffusion_extractor = DiffusionExtractor(config_run, "cuda")
dims = collect_dims(diffusion_extractor.unet, idxs=diffusion_extractor.idxs)
aggregation_network = AggregationNetwork(
    descriptor_size=config_run.model.descriptor_size,
    feature_dims=dims,
    device="cuda",
)
model = DiffusionFeatureExtractor(
    config=config_run,
    threshold=0.2,
    diffusion_extractor=diffusion_extractor,
    aggregation_network=aggregation_network,
).cuda()
weights.load_checkpoint(model=model.aggregation_network, pth_path=args.checkpoint)
im_transform = im_transform()

ids = range(1, 31)
config_loader = []
for id_obj in ids:
    config_loader.append(["test", "test_{:02d}".format(id_obj), "query", [id_obj], False])
    config_loader.append(["test", "templates_{:02d}".format(id_obj), "template", id_obj])

datasetLoader = {}
for config in config_loader:
    print("Dataset", config[0], config[1], config[2])
    save_sample_path = os.path.join(config_global.root_path,
                                    config_run.dataset.sample_path, dir_name, config[1])
    if config[2] == "query":
        loader = Tless(root_dir=config_global.root_path, split=config[0], use_augmentation=config[4],
                       list_id_obj=config[3], image_size=config_run.dataset.image_size,
                       mask_size=config_run.dataset.mask_size,
                       save_path=save_sample_path, im_transform=im_transform)
    else:
        loader = TemplatesTless(root_dir=config_global.root_path, id_obj=config[3],
                                image_size=config_run.dataset.image_size, mask_size=config_run.dataset.mask_size,
                                save_path=save_sample_path,
                                im_transform=im_transform)
    datasetLoader[config[1]] = loader
    print("---" * 20)

datasetLoader = init_dataloader(dict_dataloader=datasetLoader,
                                batch_size=config_run.train.batch_size,
                                num_workers=config_run.train.num_workers)

# Run and save prediction into a dataframe
for id_obj in tqdm(ids):
    save_prediction_obj_path = os.path.join(config_global.root_path,
                                            config_run.save_prediction_path, dir_name, "{:02d}".format(id_obj))
    prediction_npz_path = os.path.join(save_prediction_obj_path, "epoch_{:02d}.npz".format(0))
    testing_score = testing_utils.test(query_data=datasetLoader["test_{:02d}".format(id_obj)],
                                       template_data=datasetLoader["templates_{:02d}".format(id_obj)],
                                       model=model, id_obj=id_obj,
                                       save_prediction_path=prediction_npz_path,
                                       epoch=0, logger=trainer_logger)

# compute VSD metric with multiprocessing to accelerate

# # seen objects 1-18
pool = multiprocessing.Pool(processes=10)
seen_objects = range(1, 19)
list_pred_path = []
list_save_path = []
for i in seen_objects:
    pred_obj_path = os.path.join(config_global.root_path, config_run.save_prediction_path, dir_name,
                                 "{:02d}".format(i), "epoch_{:02d}.npz".format(0))
    save_dir = os.path.join(config_global.root_path, config_run.save_prediction_path, dir_name,
                            "{:02d}".format(i), "pred_pose_epoch_{:02d}.npz".format(0))
    list_pred_path.append(pred_obj_path)
    list_save_path.append(save_dir)
eval_vsd_with_index = partial(eval_vsd, list_prediction_path=list_pred_path,
                              list_save_path=list_save_path, root_dir=config_global.root_path)
start_time = time.time()
list_index = range(len(seen_objects))
mapped_values = list(tqdm(pool.imap_unordered(eval_vsd_with_index, list_index), total=len(list_index)))
finish_time = time.time()
seen_scores = []
for score in mapped_values:
    seen_scores.extend(score)
print("Final score of seen object: {}".format(np.mean(seen_scores)))
print("Total time to evaluate T-LESS on seen objects ", finish_time - start_time)

# unseen objects 19-31
unseen_objects = range(19, 31)
list_pred_path = []
list_save_path = []

for i in unseen_objects:
    pred_obj_path = os.path.join(config_global.root_path, config_run.save_prediction_path, dir_name,
                                 "{:02d}".format(i), "epoch_{:02d}.npz".format(0))
    save_dir = os.path.join(config_global.root_path, config_run.save_prediction_path, dir_name,
                            "{:02d}".format(i), "pred_pose_epoch_{:02d}.npz".format(0))
    list_pred_path.append(pred_obj_path)
    list_save_path.append(save_dir)
eval_vsd_with_index = partial(eval_vsd, list_prediction_path=list_pred_path,
                              list_save_path=list_save_path, root_dir=config_global.root_path)
start_time = time.time()
list_index = range(len(unseen_objects))
mapped_values = list(tqdm(pool.imap_unordered(eval_vsd_with_index, list_index), total=len(list_index)))
finish_time = time.time()
unseen_scores = []
for score in mapped_values:
    unseen_scores.extend(score)
print("Final score of unseen object: {}".format(np.mean(unseen_scores)))
print("Total time to evaluate T-LESS on unseen objects ", finish_time - start_time)
