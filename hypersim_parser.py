import numpy as np
import h5py
import cv2
import shutil
import os
import pandas as pd
import json
import re

from tqdm import tqdm


def isObjectUsefull(obj_name, obj_material):
    for stop_name in NAME_STOPLIST:
        if stop_name in obj_name:
            return False
    
    for stop_material in MATERIAL_STOPLIST:
        if stop_material in obj_material:
            return False

    return True


def process_frame(frame_image_path, frame_render_entity_id, 
                  metadata_node_strings, objects_path, markup_path, 
                  material_markup, output_materials):
    df = pd.read_csv(metadata_node_strings)

    def find_material_by_semantic_id(df, semantic_id):
        name_rows = df[(df['node_id'] == semantic_id) & (df['path'] == "<root>")]
        material_rows = df[(df['node_id'] == semantic_id) & (df['path'] == "<root>.material")]
        scene_name_rows = df[(df['node_id'] == semantic_id) & (df['path'] == "<root>.scene_name[1]")]
        
        name = "name"
        material = "material"
        scene_name = None
        if not name_rows.empty:
            name = name_rows['string'].iloc[0]
            if "@" in name:
                name = name.split("@")[0]

        if not material_rows.empty:
            material = material_rows['string'].iloc[0]
            if "@" in material:
                material = material.split("@")[0]
        
        if not scene_name_rows.empty:
            scene_name = scene_name_rows['string'].iloc[0]
            assert scene_name.count('/') >= 1, \
                f"Scene name format must be 'scene/scene_name/obj_id', got {scene_name}"
            scene_name = scene_name.split("/")[1]

        return name, material, scene_name

    with h5py.File(frame_render_entity_id, "r") as f: 
        render_entity_id_data = f["dataset"][:]

    render_ids = list(np.unique(render_entity_id_data))

    original_image = cv2.imread(frame_image_path)

    scene_objects = {}

    with open(output_materials, 'r', encoding='utf-8') as f:
        output_markup = json.load(f)

    for render_id in tqdm(render_ids):
        mask = (render_entity_id_data == render_id)

        obj_name, obj_material, obj_scene_name = find_material_by_semantic_id(df, render_id)

        masked_image = original_image * mask[:, :, np.newaxis]
        non_zero_indices = np.argwhere(mask)
        top_left = np.min(non_zero_indices, axis=0)
        bottom_right = np.max(non_zero_indices, axis=0)
        cropped_image = masked_image[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
        if cropped_image.shape[0] * cropped_image.shape[1] < MIN_IMAGE_AREA:
            continue

        usefull_object = isObjectUsefull(obj_name, obj_material)
        
        if not usefull_object:
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
    
        obj_filename = f"combined_obj_{scene_name}.png"
        object_path = os.path.join(objects_path, obj_filename)
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
            output_markup[obj_info['image_path']] = obj_info
            output_markup[obj_info['image_path']]['name'] = cleaned_names
            del output_markup[obj_info['image_path']]['image_path']
            cv2.imwrite(object_path, cropped_image)
        else:
            print('Skip', obj_info['image_path'])

    with open(output_materials, 'w') as f:
        json.dump(output_markup, f, indent=4)


def main():
    global OBJECT_SAVEPATH
    global NAME_STOPLIST, MATERIAL_STOPLIST
    global MIN_IMAGE_AREA

    SCENE_ROOT_PATH = "../scenes"
    OBJECT_SAVEPATH = "material_dataset_v2"
    CAM_NAME = "cam_00"
    
    NAME_STOPLIST = ['tile', 'window', 'wall', 'floor', 'mirror', 'list']
    MATERIAL_STOPLIST = ['wall']
    MIN_IMAGE_AREA = 40 * 40

    FRAME_COUNT = 100

    MATERIAL_MARKUP_PATH = "materials_info.json"
    with open(MATERIAL_MARKUP_PATH, 'r', encoding='utf-8') as f:
        material_markup = json.load(f)

    if os.path.exists(OBJECT_SAVEPATH):
        shutil.rmtree(OBJECT_SAVEPATH)
    os.makedirs(OBJECT_SAVEPATH)
    

    OUTPUT_MATERIALS = f'{OBJECT_SAVEPATH}/processed_materials.json'
    with open(OUTPUT_MATERIALS, 'w') as f:
        json.dump({}, f, indent=4)

    scene_names = ["ai_001_001", "ai_001_003", "ai_001_004",
                   "ai_001_005", "ai_001_007", "ai_001_008"]
    
    for scene_name in scene_names:
        print(f'Processing scene: {scene_name}')
        scene_path = os.path.join(OBJECT_SAVEPATH, scene_name)
        if os.path.exists(scene_path):
            shutil.rmtree(scene_path)
        os.makedirs(scene_path)

        markups_path = os.path.join(scene_path, "markups")
        if os.path.exists(markups_path):
            shutil.rmtree(markups_path)
        os.makedirs(markups_path)

        for frame_id in range(FRAME_COUNT):
            frame_id_str = f'{frame_id:04d}'
            print(f'\nProcessing frame: {frame_id_str}')
            
            objects_path = os.path.join(scene_path, frame_id_str)
            if os.path.exists(objects_path):
                shutil.rmtree(objects_path)
            os.makedirs(objects_path)

            markup_path = os.path.join(markups_path, f'{frame_id_str}.json')

            metadata_node_strings = f"{SCENE_ROOT_PATH}/{scene_name}/_detail/metadata_node_strings.csv"
            frame_image_path = f"{SCENE_ROOT_PATH}/{scene_name}/images/scene_{CAM_NAME}_final_preview/frame.{frame_id_str}.color.jpg"
            frame_render_entity_id = f"{SCENE_ROOT_PATH}/{scene_name}/images/scene_{CAM_NAME}_geometry_hdf5/frame.{frame_id_str}.render_entity_id.hdf5"

            if not os.path.exists(frame_render_entity_id):
                print(f'Skipping frame {frame_id_str}')
                continue

            process_frame(frame_image_path, frame_render_entity_id, 
                          metadata_node_strings, objects_path, markup_path,
                          material_markup, OUTPUT_MATERIALS)

if __name__ == "__main__":
    main()
