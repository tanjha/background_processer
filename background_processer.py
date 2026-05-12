from PIL import Image as pillow
from PIL import ImageColor
from pathlib import Path as path

import tkinter as tk
from tkinter import filedialog
import time

import biref_process

# Get Information -> Process VIA rembg + Save -> Process VIA pillow + Save
# Information: input dir, rembg output dir, pillow output dir

root_dir = ""
input_directory_path = ""
transparent_directory_path = ""
background_directory_path = ""
save_drive = False
paste_background = False 
background_hex = ""


def main():
    get_information()
    process()
    print("Finished")


def get_information():
    global root_dir, input_directory_path, transparent_directory_path
    global background_directory_path, save_drive, paste_background
    global background_hex

    root = tk.Tk()
    root.withdraw()  # Hide the root window

    print("-------------------- Program Parameters --------------------")

    do_background = input("Do you want to put a background on images? ('yes' or 'no'): ")
    if do_background == 'yes':
        background_hex = input("What color do you want the background to be? (hex code): ")
        paste_background = True
    

    print("-------------------- Directory Inputs --------------------")
    print("Select your input directory...")
    time.sleep(1)
    input_directory_path = filedialog.askdirectory(title="Select Input Directory")


    print("Select model output directory...")
    time.sleep(1)
    transparent_directory_path = filedialog.askdirectory(title="Select Transparent Output Directory")
    
    if paste_background:
        print("Select background output directory...")
        time.sleep(1)
        background_directory_path = filedialog.askdirectory(title="Select Background Output Directory")

    print("-------------------- Other Information --------------------")
    save_google = input("Do you want to save to google drive? ('yes' or 'no'): ")
    if save_google == 'yes': save_drive = True
    
    

def process():
    # Process images through model
    biref_process.process_images(input_directory_path, transparent_directory_path)

    print("Finished model processing, starting background processing")

    # Glob after model has written output files
    open_dir = path(transparent_directory_path).glob('*.png')

    for img_path in open_dir:
        #Add Background colour
        if paste_background:
            img = pillow.open(img_path).convert("RGBA")
            
            bgColor = ImageColor.getrgb(background_hex) + (255,)
    

            new_img_data = []
            for pixel in img.getdata():
                if pixel[3] == 0:  # Transparent pixel
                    new_img_data.append(bgColor)
                else:
                    new_img_data.append(pixel)

            img.putdata(new_img_data) 

            save_path = path(background_directory_path) / img_path.name
            img.save(save_path, "PNG")


        print(f"Processed {img_path}")
    print("Done")



if __name__ == '__main__':
    main()