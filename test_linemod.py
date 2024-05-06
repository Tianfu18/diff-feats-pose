import argparse
import os

from lib.utils import weights, logger
from lib.utils.config import Config

from lib.models.diffusion.stable_diffusion.resnet import collect_dims
from lib.models.diffusion.diffusion_extractor import DiffusionExtractor
from lib.models.diffusion.aggregation_network import AggregationNetwork
from lib.models.diffusion_network import DiffusionFeatureExtractor
from lib.datasets.dataloader_utils import init_dataloader

from lib.datasets.linemod.dataloader_query import LINEMOD
from lib.datasets.linemod.dataloader_template import TemplatesLINEMOD
from lib.datasets.im_transform import im_transform, tensor2im

from lib.datasets.linemod import inout
from lib.datasets.linemod import testing_utils

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, choices=['split1', 'split2', 'split3'])
parser.add_argument('--config_path', type=str)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--result_path', type=str, default='./dataset/results/vis/linemod')
args = parser.parse_args()

config_global = Config(config_file="./config.json").get_config()
config_run = Config(args.config_path).get_config()

args.result_path = os.path.join(args.result_path, config_run.dataset.split)
if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)

# initialize global config for the training
dir_name = (args.config_path.split('/')[-1]).split('.')[0]
print("config", dir_name)
save_path = os.path.join(config_global.root_path, config_run.log.weights, dir_name)
trainer_dir = os.path.join(os.getcwd(), "logs")
trainer_logger = logger.init_logger(save_path=save_path,
                                    trainer_dir=trainer_dir,
                                    trainer_logger_name=dir_name)


# initialize network
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
tensor2im = tensor2im


# create dataloader for query wo occlusion: train_loader, (test_seen_loader, test_unseen_loader)
# query with occlusion: (test_seen_occ_loader, test_unseen_occ_loader),
# template: (template_loader,  template_unseen_loader)
seen_id_obj, seen_names, seen_occ_id_obj, seen_occ_names, unseen_id_obj, unseen_names, \
unseen_occ_id_obj, unseen_occ_names = inout.get_list_id_obj_from_split_name(config_run.dataset.split)
config_loader = [["seen_test", "seen_test", "LINEMOD", seen_id_obj],
                 ["unseen_test", "test", "LINEMOD", unseen_id_obj],
                 ["seen_template", "test", "templatesLINEMOD", seen_id_obj],
                 ["unseen_template", "test", "templatesLINEMOD", unseen_id_obj],

                 ["seen_occ_test", "test", "occlusionLINEMOD", seen_occ_id_obj],
                 ["unseen_occ_test", "test", "occlusionLINEMOD", unseen_occ_id_obj],
                 ["seen_occ_template", "test", "templatesLINEMOD", seen_occ_id_obj],
                 ["unseen_occ_template", "test", "templatesLINEMOD", unseen_occ_id_obj]]

datasetLoader = {}
for config in config_loader:
    print("Dataset", config[0], config[2], config[3])
    save_sample_path = os.path.join(config_global.root_path, config_run.dataset.sample_path, dir_name,
                                    config[0])
    if config[2] == "templatesLINEMOD":
        loader = TemplatesLINEMOD(root_dir=config_global.root_path, dataset=config[2], list_id_obj=config[3],
                                  split=config[1], image_size=config_run.dataset.image_size,
                                  mask_size=config_run.dataset.mask_size, im_transform=im_transform,
                                  save_path=save_sample_path)
    else:
        loader = LINEMOD(root_dir=config_global.root_path,
                         dataset=config[2], list_id_obj=config[3], split=config[1],
                         image_size=config_run.dataset.image_size,
                         mask_size=config_run.dataset.mask_size, im_transform=im_transform,
                         save_path=save_sample_path)
    datasetLoader[config[0]] = loader
    print("---" * 20)

datasetLoader = init_dataloader(dict_dataloader=datasetLoader,
                                batch_size=config_run.train.batch_size,
                                num_workers=config_run.train.num_workers)

new_score = {}
for config_split in [["seen", seen_id_obj], ["seen_occ", seen_occ_id_obj],
                     ["unseen", unseen_id_obj], ["unseen_occ", unseen_occ_id_obj]]:
    query_name = config_split[0] + "_test"
    template_name = config_split[0] + "_template"
    testing_score = testing_utils.test(query_data=datasetLoader[query_name],
                                       template_data=datasetLoader[template_name],
                                       model=model, split_name=config_split[0],
                                       list_id_obj=config_split[1].tolist(), epoch=0,
                                       logger=trainer_logger,
                                       vis=False,
                                       result_vis_path=args.result_path,
                                       tensor2im=tensor2im)
    new_score[config_split[0] + "_err"] = testing_score[0]
    new_score[config_split[0] + "_acc"] = testing_score[-1]
print(new_score)