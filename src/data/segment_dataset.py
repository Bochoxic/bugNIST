from skimage import io

import matplotlib.pyplot as plt
import numpy as np

import cv2
import os
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def threshold(image, kernel, threshold_n):
    mask = image
    T, mask = cv2.threshold(mask, threshold_n, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def prepare_data(raw_directory, processed_directory):
    directories = os.listdir(raw_directory)
    kernel = np.ones((9,9),np.uint8)
    threshold_n = 90
    max_images = 10


    for directory in directories:
        volumens = os.listdir(raw_directory+"/"+directory)

        save_dir_mask_volume = processed_directory + "/" + directory + "_masks/"
        save_dir_mask = processed_directory + "/masks/"
        save_dir_imgs = processed_directory + "/imgs/"

        if os.path.exists(save_dir_mask) == False:
            print(f"Directory {save_dir_mask} not found, creating it...")
            os.mkdir(save_dir_mask)

        if os.path.exists(save_dir_imgs) == False:
            print(f"Directory {save_dir_imgs} not found, creating it...")
            os.mkdir(save_dir_imgs)
        
        if os.path.exists(save_dir_mask_volume) == False:
            print(f"Directory {save_dir_mask_volume} not found, creating it...")
            os.mkdir(save_dir_mask_volume)

        number_images = 0
        for volume in volumens:
            if number_images > max_images:
                print("Number of max images allowed")
                break
            number_images = number_images + 1
            print(f"Directory: {directory}\t Image: {volume}\t Number={number_images}")
            in_dir = raw_directory + "/" + directory + "/"
            image_tif = io.imread(in_dir+volume)
            mask_volume = np.empty_like(image_tif)
            for i in range(0, image_tif.shape[0]):
                mask = threshold(image_tif[i], kernel, threshold_n)
                mask_volume[i,:,:] = mask
                image_name = volume[:-4]
                io.imsave(save_dir_imgs+image_name+str(i)+".png", image_tif[i])
                io.imsave(save_dir_mask+image_name+str(i)+".png",mask)


            io.imsave(save_dir_mask_volume+volume, mask_volume)




def visualize_mask():
    in_dir = "data/raw/OriginalScans/Bank 1/"
    im_name = "Bank 1_000.tif"
    image_tif = io.imread(in_dir + im_name)

    Tot = int(image_tif.shape[0] / 50)*2
    Cols = 2
    Rows = Tot // Cols
    if Tot % Cols != 0:
        Rows += 1   
    Position = range(1,Tot + 1)

    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(1, figsize=(720*px,1080*px))
    kernel = np.ones((9,9),np.uint8)
    threshold_n = 90

    for k in range(Tot):

        ax = fig.add_subplot(Rows,Cols,Position[k])
        if k % 2:
            ax.imshow(image_tif[int(k*50/2)])
        else:
            mask = threshold(image_tif[int(k*50/2)], kernel, threshold_n)
            ax.imshow(mask)


    # Threshold 90, closing 10
    plt.show()

def main():
    raw_directory = "data/raw/OriginalScans"
    processed_directory = "data/processed"

    prepare_data(raw_directory, processed_directory)

if __name__ == "__main__":
    main()