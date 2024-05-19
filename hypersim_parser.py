import csv
import concurrent.futures
import json
import os
import re
import shutil

import cv2
import h5py
import numpy as np
import pandas as pd

from collections import defaultdict
from tqdm import tqdm


def parse_hierarchy_data(file_path):
    data = defaultdict(dict)

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for node_id, key_path, value in reader:
            current_dict = data[int(node_id)]
            assert key_path not in current_dict, f'Key {key_path} already in data for node {node_id}'
            current_dict[key_path] = value
    
    return data

def get_object_attributes(data, semantic_id):
    object_info = data[semantic_id]
    
    try:
        object_name = object_info['<root>']
    except Exception as e:
        print(object_info, semantic_id)
        raise e
    if "@" in object_name:
        object_name = object_name.split("@")[0]
    
    if '<root>.material.scene_name[0]' in object_info:
        object_material = object_info['<root>.material.scene_name[0]']
    elif '<root>.material.base_material' in object_info:
        object_material = object_info['<root>.material.base_material']
    else:
        object_material = object_info['<root>.material']
    
    if "@" in object_material:
        object_material = object_material.split("@")[0]

    object_scene_name = object_info['<root>.scene_name[1]']
    object_scene_name = object_scene_name.split("/")[1]

    return object_name, object_material, object_scene_name


def isObjectUsefull(obj_name, obj_material, name_stoplist, material_stoplist):
    for stop_name in name_stoplist:
        if stop_name in obj_name:
            return False
    
    for stop_material in material_stoplist:
        if stop_material in obj_material:
            return False

    return True


def process_frame(frame_image_path, frame_render_entity_id, 
                  objects_path, object_infos_dict, 
                  name_stoplist, material_stoplist, 
                  material_markup, scene_materials_dict, min_image_area):

    with h5py.File(frame_render_entity_id, "r") as f: 
        render_entity_id_data = f["dataset"][:]

    render_ids = list(np.unique(render_entity_id_data))

    original_image = cv2.imread(frame_image_path)

    # Skip black images
    if np.all(original_image == 0):
        return

    scene_objects = {}

    for render_id in render_ids:
        if render_id == -1:
            continue

        obj_name, obj_material, obj_scene_name = get_object_attributes(object_infos_dict, render_id)

        usefull_object = isObjectUsefull(obj_name, obj_material, name_stoplist, material_stoplist)
        if not usefull_object:
            continue

        mask = (render_entity_id_data == render_id)
        non_zero_indices = np.argwhere(mask)
        top_left = np.min(non_zero_indices, axis=0)
        bottom_right = np.max(non_zero_indices, axis=0)

        w = bottom_right[0] - top_left[0]
        h = bottom_right[1] - top_left[1]
        crop_area = (bottom_right[0] - top_left[0]) * (bottom_right[1] - top_left[1])
        if crop_area < min_image_area \
            or h < (min_image_area ** 0.5) / 2 \
            or w < (min_image_area ** 0.5) / 2:
            continue

        if obj_scene_name not in scene_objects:
            scene_objects[obj_scene_name] = {
                'name': [obj_name],
                'material': set([obj_material]),
                'mask': mask[:, :, np.newaxis]
            }
        else:
            scene_objects[obj_scene_name]['name'].append(obj_name)
            scene_objects[obj_scene_name]['material'].add(obj_material)
            scene_objects[obj_scene_name]['mask'] += mask[:, :, np.newaxis]

    for scene_name, obj_info in scene_objects.copy().items():
        masked_image = original_image * obj_info["mask"]
        non_zero_indices = np.argwhere(obj_info["mask"])
        top_left = np.min(non_zero_indices, axis=0)
        bottom_right = np.max(non_zero_indices, axis=0)
        cropped_image = masked_image[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]

        # objects_path has a format path_to_scene/frame_id
        object_path = f"{objects_path}_{scene_name}.png"
        obj_info['image_path'] = object_path
        
        cleaned_materials = [re.sub(r'[\d_]+', '', m).strip().lower() for m in obj_info["material"]]
        cleaned_names = list(set([re.sub(r'[\d_]+', '', re.sub(r'obj', '', m)).strip().lower() for m in obj_info["name"]]))

        relevant_materials = set()
        for material in cleaned_materials:
            for category, items in material_markup.items():
                if any(item.lower() in material for item in items):
                    relevant_materials.add(category)

        obj_info["material"] = list(relevant_materials)
        del obj_info["mask"]

        if len(obj_info["material"]) > 0:
            scene_name_appeared_count = scene_materials_dict['object_counter'].get(scene_name, 0)
            if scene_name_appeared_count < 20:
                scene_materials_dict[obj_info['image_path']] = obj_info
                scene_materials_dict[obj_info['image_path']]['name'] = cleaned_names
                del scene_materials_dict[obj_info['image_path']]['image_path']
                cv2.imwrite(object_path, cropped_image)
                scene_materials_dict['object_counter'][scene_name] = scene_name_appeared_count + 1


def process_scene(scene_name, material_markup, 
                  scenes_root_path, output_dataset_path, 
                  name_stoplist, material_stoplist,
                  min_image_area):
    scene_output_path = os.path.join(output_dataset_path, scene_name)
    if os.path.exists(scene_output_path):
        shutil.rmtree(scene_output_path)
    os.makedirs(scene_output_path)

    scene_materials_dict = {}

    if not os.path.exists(scene_output_path):
        print(f'Skip: {scene_output_path}')
        return scene_materials_dict
    
    scene_materials_dict['object_counter'] = {}

    camera_list = [path for path in os.listdir(f"{scenes_root_path}/{scene_name}/_detail")
                    if "cam_" in path]
    for camera_name in sorted(camera_list):
        camera_image_path = f"{scenes_root_path}/{scene_name}/images/scene_{camera_name}_final_preview"
        camera_markup_path = f"{scenes_root_path}/{scene_name}/images/scene_{camera_name}_geometry_hdf5"
        metadata_node_strings = f"{scenes_root_path}/{scene_name}/_detail/metadata_node_strings.csv"

        camera_frames = [filename for filename in os.listdir(camera_image_path) 
                        if ".tonemap.jpg" in filename]
        
        for frame_filename in camera_frames:
            frame_id_str = frame_filename.replace("frame.", "").replace(".tonemap.jpg", "")
            
            objects_path = os.path.join(scene_output_path, f'{camera_name}_{frame_id_str}')

            frame_image_path = f"{camera_image_path}/frame.{frame_id_str}.tonemap.jpg"
            frame_render_entity_id = f"{camera_markup_path}/frame.{frame_id_str}.render_entity_id.hdf5"

            if not os.path.exists(frame_render_entity_id):
                print(f'Skipping frame {frame_id_str}')
                continue

            object_infos_dict = parse_hierarchy_data(metadata_node_strings)

            process_frame(
                frame_image_path, frame_render_entity_id, 
                objects_path, object_infos_dict,
                name_stoplist, material_stoplist,
                material_markup, scene_materials_dict, min_image_area
            )

    del scene_materials_dict['object_counter']
    
    return scene_materials_dict


def main():
    SCENES_ROOT_PATH = "../scenes"
    OUTPUT_DATASET_PATH = "material_dataset_135_scenes"
    
    NAME_STOPLIST = ['tile', 'window', 'wall', 'floor', 'mirror', 'list']
    MATERIAL_STOPLIST = ['wall']
    MIN_IMAGE_AREA = 50 * 50

    MATERIAL_MARKUP_PATH = "materials_info_v2.json"
    with open(MATERIAL_MARKUP_PATH, 'r', encoding='utf-8') as f:
        material_markup = json.load(f)

    if os.path.exists(OUTPUT_DATASET_PATH):
        print(f'Dataset {OUTPUT_DATASET_PATH} exists!')
        exit(0)
    os.makedirs(OUTPUT_DATASET_PATH)
    

    OUTPUT_MATERIALS = f'{OUTPUT_DATASET_PATH}/processed_materials.json'

    dataset_markup = {}
    with open(OUTPUT_MATERIALS, 'w') as f:
        json.dump(dataset_markup, f, indent=4)

    scenes = sorted(os.listdir(SCENES_ROOT_PATH))
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(16, os.cpu_count())) as executor:
        futures = {executor.submit(process_scene, scene_name, material_markup, 
                                   SCENES_ROOT_PATH, OUTPUT_DATASET_PATH, 
                                   NAME_STOPLIST, MATERIAL_STOPLIST, 
                                   MIN_IMAGE_AREA): scene_name for scene_name in scenes}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(scenes), desc="Processing scenes"):
            scene_name = futures[future]
            try:
                output_scene_materials = future.result()
                for k, v in output_scene_materials.items():
                    dataset_markup[k] = v
                with open(OUTPUT_MATERIALS, 'w') as f:
                    json.dump(dataset_markup, f, indent=4)
            except Exception as exc:
                print(f'{scene_name} generated an exception: {exc}')


if __name__ == "__main__":
    main()
