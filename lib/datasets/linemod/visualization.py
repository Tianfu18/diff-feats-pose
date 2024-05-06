import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def visualize_gt_templates(crop_dir, dataset, obj_name, idx_frame, idx_test_template, idx_train_template, test_error,
                           train_error):
    save_dir = os.path.join(crop_dir, "visualization_linemod", dataset, obj_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    query_img = Image.open(os.path.join(crop_dir, dataset, obj_name, '{:06d}.png'.format(idx_frame)))
    test_template = Image.open(os.path.join(crop_dir, "templatesLINEMOD", "test", obj_name,
                                            '{:06d}.png'.format(idx_test_template)))
    train_template = Image.open(os.path.join(crop_dir, "templatesLINEMOD", "train", obj_name,
                                             '{:06d}.png'.format(idx_train_template)))
    plt.figure(figsize=(5, 15))
    plt.subplot(1, 3, 1)
    plt.imshow(query_img)
    plt.axis('off')
    plt.title("Query")

    plt.subplot(1, 3, 2)
    plt.imshow(test_template)
    plt.axis('off')
    plt.title("Test, Err={:.2f}".format(test_error))

    plt.subplot(1, 3, 3)
    plt.imshow(train_template)
    plt.axis('off')
    plt.title("Train, Err={:.2f}".format(train_error))

    plt.savefig(os.path.join(save_dir, "{:06d}.png".format(idx_frame)), bbox_inches='tight', dpi=100)
    plt.close("all")


def visualize_result(query, template, mask, score_matrix, id_obj, score, gt_pose, save_path):
    img_size = query.shape[0]
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    outline = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1) - mask
    outline = cv2.dilate(outline, np.ones((3, 3), np.uint8), iterations=1)

    score_size = score_matrix.shape[0]
    color_score_img = np.zeros((score_size, score_size, 3))
    # negative to red and positive to green
    plus_score_img, neg_score_img = score_matrix.copy(), score_matrix.copy()
    plus_score_img[plus_score_img < 0] = 0
    neg_score_img[neg_score_img > 0] = 0
    color_score_img[:, :, 1] = plus_score_img
    color_score_img[:, :, 2] = -neg_score_img
    color_score_img = cv2.resize(
        color_score_img, (img_size, img_size), interpolation=cv2.INTER_AREA
    ) * 255.
    color_score_img = color_score_img.astype(np.uint8)
    color_score_img[mask != 1] = [0, 0, 0]
    color_score_img[outline == 1] = [255, 255, 255]
    pose_result = query.copy()
    pose_result[outline == 1] = [0, 0, 255]

    dir_name = f'{score * 100}_{id_obj}'

    if not os.path.exists(os.path.join(save_path, str(id_obj), dir_name)):
        os.makedirs(os.path.join(save_path, str(id_obj), dir_name))

    cv2.imwrite(os.path.join(save_path, str(id_obj), dir_name, 'query.png'), query)
    cv2.imwrite(os.path.join(save_path, str(id_obj), dir_name, 'template.png'), template)
    cv2.imwrite(os.path.join(save_path, str(id_obj), dir_name, 'score.png'), color_score_img)
    cv2.imwrite(os.path.join(save_path, str(id_obj), dir_name, 'pose_result.png'), pose_result)
