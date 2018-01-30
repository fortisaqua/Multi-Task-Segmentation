import os
import glob
import SimpleITK as ST
import sys

def read_dicoms(input_directory):
    if len(sys.argv)<1:
        print "Usage: DicomSeriesReader <input_directory> <output_file>"
        sys.exit(1)

    # print "Reading Dicom directory",input_directory
    reader=ST.ImageSeriesReader()

    dicom_names=reader.GetGDCMSeriesFileNames(input_directory)
    reader.SetFileNames(dicom_names)
    # print dicom_names

    image=reader.Execute()
    return image

def organizie_keys(data_root):
    ret = []
    for project_name in os.listdir(data_root):
        project_dir = data_root+'/'+project_name
        project_meta = {}
        project_meta['project_name']=project_name
        for object in os.listdir(project_dir):
            project_meta[str(object)] = project_dir+'/'+object
        ret.append(project_meta)
    return ret

def split_metas(original_meta):
    ret = []
    for item in original_meta:
        keys = item.keys()
        origins = []
        masks = {}
        for name in keys:
            if "original" in name:
                origins.append(item[name])
            else:
                masks[name] = item[name]
        for original_dicom_dir in origins:
            sub_meta={}
            sub_meta["original"]=original_dicom_dir
            for mask_name in masks.keys():
                sub_meta[mask_name] = masks[mask_name]
            ret.append(sub_meta)
    return ret

def get_records(record_dir):
    pattern = record_dir+'data_set_*.tfrecord'
    tfrecords = glob.glob(pattern)
    return tfrecords