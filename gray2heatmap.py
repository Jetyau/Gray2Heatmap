# Convert grayscale img to heatmap according to the discussion on https://stackoverflow.com/questions/59478962/how-to-convert-a-grayscale-image-to-heatmap-image-with-python-opencv
# load csv (Support 2D only) and convert to png heatmap


import cv2
import argparse
import numpy as np
from glob import glob
import pandas as pd
import os
import time


def cvt2heatmap(img_rgb, img_gray, heatmap_type):
    
    img_gray = img_gray[:,1:]
    img_rgb = cv2.resize(img_rgb, (256,256))
    img_gray = (img_gray-0.5)*10 *255 # Autosetting?
    img_gray = img_gray.clip(min=0, max=255)
    img_gray = cv2.GaussianBlur(img_gray,(3,3),0)
    img_gray=img_gray.astype('uint8')
    heatmap_type =getattr(cv2, heatmap_type) #e.g.: getattr(cv2, 'COLORMAP_JET')
    heatmap = cv2.applyColorMap(img_gray, heatmap_type)
    


    blend = (0.5 * img_rgb + 0.5*heatmap).astype(np.uint8)
    return blend, heatmap
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default='', type=str, help='path to img folder')
    parser.add_argument('-he','--heatmaptype', default='cv2.COLORMAP_JET', type=str, help='select heatmap')
    parser.add_argument('-fe', '--fetch_origin_img', default=True, help='fetch origin img to heatmap folder?')
    args = parser.parse_args()

    # get setting
    path = args.path
    
    heatmap_type = args.heatmaptype
    fetch_origin_img = args.fetch_origin_img
    rgb_root = ''
    


    # get img path
    img_list = glob(path + '/*.csv')
    if not os.path.exists(path+'/heatmaps'):
        os.mkdir(path+'/heatmaps')
    save_dir = path+'/heatmaps/'

    for _, img_path in enumerate(img_list):
        junk = img_path.split('/')
        if 'log.csv' not in junk[-1]:
            gray_img_df = pd.read_csv(img_path)
            gray_img_npy = gray_img_df.to_numpy()
            
            if fetch_origin_img:
                # ref uncertainty_map_case00002movieframemovieFrame_062058.png_138.csv
                # ref uncertainty_map_case11movieframemovieFrame_068440.png_54.csv
                # ref uncertainty_map_case6movieframemovieFrame_256740.png_283.csv
                
                img_name = junk[-1][(junk[-1].find('movieframe')+10):junk[-1].find('png')+3]
                case_dir = junk[-1][junk[-1].find('case'):junk[-1].find('movieframe')]
                epoch = junk[-1][junk[-1].find('png_')+4:junk[-1].find('.csv')]
                junk_dir = 'movieframe'
                img_rgb_path = os.path.join(rgb_root, case_dir, junk_dir, img_name)
                img_rgb = cv2.imread(img_rgb_path)
                blend_img, heatmap = cvt2heatmap(img_rgb, gray_img_npy, heatmap_type)
                
                to_save_path_blend = save_dir + "epoch{}".format(epoch) + img_name
                
                
                
                
                to_save_path_heatmap = save_dir + 'heatmap_' + "epoch{}".format(epoch) + img_name
                cv2.imwrite(to_save_path_blend, blend_img)
                cv2.imwrite(to_save_path_heatmap, heatmap)











if __name__ == "__main__":
    main()