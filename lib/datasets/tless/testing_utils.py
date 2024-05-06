import time
import torch
import numpy as np
from lib.datasets.tless.inout import save_results
from tqdm import tqdm


def tensor2im(tensor: torch.Tensor):
    tensor = tensor / 2. + .5
    im = (tensor * 255.).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[:, :, :, ::-1]
    return im


def test(query_data, template_data, model, epoch, logger, id_obj, save_prediction_path):
    print("Testing object {}".format(id_obj))
    start_time = time.time()

    query_size, query_dataloader = len(query_data), iter(query_data)
    template_size, template_dataloader = len(template_data), iter(template_data)
    timing_text = "Validation time for epoch {}: {:.02f} minutes"

    model.eval()
    with torch.no_grad():
        list_feature_template, list_synthetic_pose, list_mask, list_idx_template, list_inplane = [], [], [], [], []
        list_img_template = []

        for i in tqdm(range(template_size)):
            # read all templates and its poses
            miniBatch = template_dataloader.next()

            template = miniBatch["template"].cuda()
            obj_pose = miniBatch["obj_pose"].cuda()
            idx_template = miniBatch["idx_template"].cuda()
            inplane = miniBatch["inplane"].cuda()
            mask = miniBatch["mask"].cuda().float()
            feature_template = model(template).cuda()

            list_synthetic_pose.append(obj_pose.cuda())
            list_mask.append(mask.cuda())
            list_feature_template.append(feature_template.cuda())
            list_idx_template.append(idx_template.cuda())
            list_inplane.append(inplane.cuda())
            list_img_template.append(tensor2im(template))

        list_feature_template = torch.cat(list_feature_template, dim=0)
        list_synthetic_pose = torch.cat(list_synthetic_pose, dim=0)
        list_mask = torch.cat(list_mask, dim=0)
        list_idx_template = torch.cat(list_idx_template, dim=0)
        list_inplane = torch.cat(list_inplane, dim=0)
        list_img_template = np.concatenate(list_img_template, axis=0)

        names = ["obj_pose", "id_obj", "id_scene", "id_frame", "idx_frame", "idx_obj_in_scene", "visib_fract",
                 "gt_idx_template", "gt_inplane",
                 "pred_template_pose", "pred_idx_template", "pred_inplane"]
        results = {names[i]: [] for i in range(len(names))}
        for i in tqdm(range(query_size)):
            miniBatch = query_dataloader.next()

            query = miniBatch["query"].cuda()
            feature_query = model(query).cuda()

            # get best template
            matrix_sim, _ = model.calculate_similarity_for_search(feature_query, list_feature_template, list_mask,
                                                               training=False)
            weight_sim, pred_index = matrix_sim.topk(k=1)
            pred_template_pose = list_synthetic_pose[pred_index.reshape(-1)]
            pred_idx_template = list_idx_template[pred_index.reshape(-1)]
            pred_inplane = list_inplane[pred_index.reshape(-1)]

            pred_template = list_img_template[pred_index.reshape(-1).cpu().numpy()]

            for name in names[:-3]:
                data = miniBatch[name].detach().numpy()
                results[name].extend(data)
            results["pred_template_pose"].extend(pred_template_pose.cpu().detach().numpy())
            results["pred_idx_template"].extend(pred_idx_template.cpu().detach().numpy())
            results["pred_inplane"].extend(pred_inplane.cpu().detach().numpy())

    save_results(results, save_prediction_path)
    logger.info("Prediction of epoch {} of object {} is saved at {}".format(epoch, id_obj, save_prediction_path))
    logger.info(timing_text.format(epoch, (time.time() - start_time) / 60))