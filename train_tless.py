import argparse
import os
import torch
from tqdm import tqdm

from lib.utils import weights, metrics, logger
from lib.utils.config import Config
from lib.datasets.dataloader_utils import init_dataloader
from lib.utils.optimizer import adjust_learning_rate

from lib.models.diffusion.diffusion_extractor import DiffusionExtractor
from lib.models.diffusion.aggregation_network import AggregationNetwork
from lib.models.diffusion.stable_diffusion.resnet import collect_dims
from lib.models.diffusion_network import DiffusionFeatureExtractor

from lib.datasets.tless.dataloader_query import Tless
from lib.datasets.tless.dataloader_template import TemplatesTless
from lib.datasets.tless import training_utils, testing_utils
from lib.datasets.im_transform import im_transform

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str)
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
im_transform = im_transform()

seen_ids, unseen_ids = range(1, 18), range(19, 31)
config_loader = [["train", "train", "query", seen_ids, config_run.dataset.use_augmentation]]
for id_obj in unseen_ids:
    config_loader.append(["test", "test_{:02d}".format(id_obj), "query", [id_obj], False])
    config_loader.append(["test", "templates_{:02d}".format(id_obj), "template", id_obj])

datasetLoader = {}
for config in config_loader:
    print("Dataset", config[0], config[1], config[2])
    save_sample_path = os.path.join(config_global.root_path,
                                    config_run.dataset.sample_path, dir_name, config[1])
    if config[2] == "query":
        loader = Tless(root_dir=config_global.root_path, split=config[0], use_augmentation=config[4],
                       list_id_obj=config[3],
                       image_size=config_run.dataset.image_size, save_path=save_sample_path,
                       mask_size=config_run.dataset.mask_size,
                       im_transform=im_transform)
    else:
        loader = TemplatesTless(root_dir=config_global.root_path, id_obj=config[3],
                                image_size=config_run.dataset.image_size, mask_size=config_run.dataset.mask_size,
                                save_path=save_sample_path, im_transform=im_transform)
    datasetLoader[config[1]] = loader
    print("---" * 20)

datasetLoader = init_dataloader(dict_dataloader=datasetLoader,
                                batch_size=config_run.train.batch_size,
                                num_workers=config_run.train.num_workers)

# initialize optimizer
optimizer = torch.optim.Adam(
    list(model.parameters()),
    lr=config_run.train.optimizer.lr,
    weight_decay=config_run.train.optimizer.weight_decay
)
scores = metrics.init_score()

for epoch in tqdm(range(0, config_run.train.epochs)):
    # update learning rate
    if epoch in config_run.train.scheduler.milestones:
        adjust_learning_rate(optimizer, config_run.train.optimizer.lr, config_run.train.scheduler.gamma)

    if epoch % 5 == 0 and epoch > 0:
        for id_obj in unseen_ids:
            save_prediction_obj_path = os.path.join(config_run.save_prediction_path, dir_name, "{:02d}".format(id_obj))
            testing_score = testing_utils.test(query_data=datasetLoader["test_{:02d}".format(id_obj)],
                                               template_data=datasetLoader["templates_{:02d}".format(id_obj)],
                                               model=model, id_obj=id_obj,
                                               save_prediction_path=os.path.join(config_global.root_path,
                                                                                 save_prediction_obj_path,
                                                                                 "epoch_{:02d}".format(epoch)),
                                               epoch=epoch,
                                               logger=trainer_logger)

    train_loss = training_utils.train(train_data=datasetLoader["train"],
                                      model=model, optimizer=optimizer,
                                      warm_up_config=[1000, config_run.train.optimizer.lr],
                                      epoch=epoch, logger=trainer_logger,
                                      log_interval=config_run.log.log_interval)

    text = '\nEpoch-{}: train_loss={} \n\n'
    weights.save_checkpoint(model, os.path.join(save_path, 'model_epoch{}.pth'.format(epoch)))
