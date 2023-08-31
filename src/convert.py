# https://www.kaggle.com/datasets/javidtheimmortal/high-resolution-semantic-change-detection-dataset

import glob
import os
import shutil
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from dotenv import load_dotenv
from PIL import Image
from supervisely.io.fs import (
    dir_exists,
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from tqdm import tqdm

import src.settings as s


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # project_name = "HRSCD"
    dataset_path = "/mnt/d/datasetninja/hrscd"
    images_path = "/mnt/d/datasetninja/hrscd/images_2012/2012"
    masks_path = "/mnt/d/datasetninja/hrscd/labels_land_cover_2012/2012"
    images_ext = ".tif"
    batch_size = 3
    ds_name = "ds"

    def create_ann(image_path):
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        mask_path = os.path.join(curr_masks_path, get_file_name_with_ext(image_path))

        if file_exists(mask_path):
            tif_data = Image.open(mask_path)
            mask_np = np.array(tif_data)
            unique_pixels = np.unique(mask_np)
            for curr_pixel in unique_pixels:
                obj_class = idx_to_class.get(int(curr_pixel))
                if obj_class is not None:
                    mask = mask_np == curr_pixel
                    ret, curr_mask = connectedComponents(mask.astype("uint8"), connectivity=8)
                    for i in range(1, ret):
                        obj_mask = curr_mask == i
                        curr_bitmap = sly.Bitmap(obj_mask)
                        if curr_bitmap.area > 50:
                            curr_label = sly.Label(curr_bitmap, obj_class)
                            labels.append(curr_label)

        tag_name = image_path.split("/")[-2]
        tags = [sly.Tag(tag_meta) for tag_meta in tag_metas if tag_meta.name == tag_name]

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    obj_class_no = sly.ObjClass("no information", sly.Bitmap)
    obj_class_surfaces = sly.ObjClass("artificial surfaces", sly.Bitmap)
    obj_class_areas = sly.ObjClass("agricultural areas", sly.Bitmap)
    obj_class_forests = sly.ObjClass("forests", sly.Bitmap)
    obj_class_wetlands = sly.ObjClass("wetlands", sly.Bitmap)
    obj_class_water = sly.ObjClass("water", sly.Bitmap)

    idx_to_class = {
        0: obj_class_no,
        1: obj_class_surfaces,
        2: obj_class_areas,
        3: obj_class_forests,
        4: obj_class_wetlands,
        5: obj_class_water,
    }

    tag_names = os.listdir(images_path)

    tag_metas = [sly.TagMeta(name, sly.TagValueType.NONE) for name in tag_names]

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)

    meta = sly.ProjectMeta(obj_classes=list(idx_to_class.values()), tag_metas=tag_metas)
    api.project.update_meta(project.id, meta.to_json())

    dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)
    progress = sly.Progress("Create dataset {}".format(ds_name), 1000)

    for zone in tag_names:
        curr_images_path = os.path.join(images_path, zone)
        curr_masks_path = os.path.join(masks_path, zone)
        if dir_exists(curr_images_path):
            images_names = os.listdir(curr_images_path)

            for images_names_batch in sly.batched(images_names, batch_size=batch_size):
                images_pathes_batch = [
                    os.path.join(curr_images_path, im_name) for im_name in images_names_batch
                ]

                img_infos = api.image.upload_paths(
                    dataset.id, images_names_batch, images_pathes_batch
                )
                img_ids = [im_info.id for im_info in img_infos]

                anns = [create_ann(image_path) for image_path in images_pathes_batch]
                api.annotation.upload_anns(img_ids, anns)

                progress.iters_done_report(len(images_names_batch))
    return project
