import os
import cv2
import argparse
import numpy as np 
import pandas as pd
import regex as re
import nibabel as nib
from skimage import measure
from skimage import io


patient_gt = []
patient_pred = []
files = []

def strip_small_lesions(target_slice, x_len, y_len):
    labels = measure.label(target_slice, background=0)
    props = pd.DataFrame(measure.regionprops_table(labels, properties=('label','bbox','axis_major_length','axis_minor_length')))
    props['remove'] = [(x<x_len) and (y<y_len) for x,y in props.loc[:,['axis_major_length','axis_minor_length']].values]
    labels_to_remove = list(props[props.remove==True].label.values)
    labels_to_remove.append(0) # adding background label to prevent background turning white
    new_labels = np.isin(labels, labels_to_remove, invert=True)
    return new_labels*1

def get_batch(filename):
    return int(re.findall("\d+", filename)[0])


def get_patient_info(filename):
    pat_id, instance, slc = re.findall("\d+", filename)
    return pat_id, instance, int(slc)


def read_file(filename, dir, strip=False):
    if filename.endswith("png"):
        image = io.imread(os.path.join(dir, filename))
        image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = (image > 0.5)*1
        if strip:
            image = strip_small_lesions(image, 3, 3)
    else:
        image = np.load(os.path.join(dir, filename))
    return image

def write_file(vol, pat_id, inst, target_dir, name):
    target_dir = os.path.join(target_dir, name)
    os.makedirs(target_dir, exist_ok = True)
    vol_name = f"subID{pat_id}_{inst}.nii.gz"
    print(f"writing :{vol_name}")
    nifti_img = nib.Nifti1Image(vol, affine=np.eye(4))
    nib.save(nifti_img, os.path.join(target_dir, vol_name))
    return vol_name


def parse_args():
    parser = argparse.ArgumentParser('Test model')
    parser.add_argument('--name', type=str, default='unet', help='Model name')
    parser.add_argument('--dataset', type=str, default='svuh', help='Dataset')
    parser.add_argument('--mode', type=str, default='pretrained', help='training mode')
    parser.add_argument('--target_csv', type=str, default='/home/prateek/from_kipchoge/ms_project/change_balcrop256/test.csv', help='Test file')
    parser.add_argument('--images_dir', type=str, default='/home/prateek/ms_change_detection/predictions/', help='Predictions directory')
    parser.add_argument('--target_dir', type=str, default='/home/prateek/ms_change_detection/stacks/', help='Output directory')
    parser.add_argument('--strip', action='store_true', help='Strip small lesions')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    target_dir = args.target_dir
    dataset = args.dataset
    mode = args.mode
    target_csv = args.target_csv
    images_dir = args.images_dir
    name = args.name
    strip = args.strip

    target_dir = os.path.join(target_dir, mode, dataset)
    images_dir = os.path.join(images_dir, mode, dataset, name)

    df = pd.read_csv(target_csv)

    files = []
    pat_list = {}
    filenames = [os.path.basename(v) for v in df.label.values]
    for i, gt_file in enumerate(filenames):
    #print(f"{i} : {gt_file}")
        gt_file = gt_file[:-1]
        pat_id, instance, slc = get_patient_info(gt_file)

        if pat_id in pat_list.keys():
            if instance in pat_list[pat_id].keys():
                pat_list[pat_id][instance]['slice'].append(slc)
                pat_list[pat_id][instance]['idx'].append(i)
            else:
                pat_list[pat_id][instance] = {'slice' : [slc], 'idx': [i]}
        else:
            pat_list[pat_id] = { instance : {'slice' : [slc], 'idx': [i]}}


    for pat_id in pat_list.keys():
        for instance in pat_list[pat_id].keys():
            slices = pat_list[pat_id][instance]['slice']
            indexs = pat_list[pat_id][instance]['idx']
            num_slices = np.max(slices)
            vol = np.zeros((256,256,num_slices+1))
            for i, slc in enumerate(slices):
                #print(f"file : test_sub-ID{pat_id}_{instance}_{slc}.png : test_{indexs[i]}")
                file = read_file(f"test_{indexs[i]}.png", images_dir, strip=strip)
                #file = read_file(f"external_{pat_id}_{instance}_{slc}.png", image_dir)
                vol[:, :, slc] = file
            vol_f = write_file(vol, pat_id, instance, target_dir, name)
            files.append({'patient': pat_id, 'instance': instance, 'file':vol_f})
    df = pd.DataFrame(files)
    df.to_csv(os.path.join(target_dir,f"{name}.csv"), index=False)