from PIL import Image as pillow
from pathlib import Path as path
from rembg import remove, new_session

# Get Information -> Process VIA rembg + Save -> Process VIA pillow + Save
# Information: input dir, rembg output dir, pillow output dir





root_dir = ""
input_directory_path = ""
transparent_directory_path = ""
background_directory_path = ""
save_drive = False
paste_background = False 
background_hex = ""
model_name = "birefnet-portrait"
session = new_session(model_name)


def main():
    get_information()
    process()
    print("Finished")


def get_information():
    print("-------------------- Directory Inputs --------------------")
    root_dir = input("Input the root directory with the 3 directories: ")
    input_directory_path = f"{root_dir}{input("Input the local directory path for the pngs (Format: .\'directory_name'): ")}"
    transparent_directory_path = f"{root_dir}{input("Input the local directory path to save the transparent images: ")}"
    background_directory_path = f"{root_dir}{input("Input the local directory path to save the images w/ backgrounds: ")}"
    print("-------------------- Other Information --------------------")
    save_google = input("Do you want to save to google drive? ('yes' or 'no'): ")
    if save_google == 'yes': save_drive = True
    do_background = input("Do you want to put a background on images? ('yes' or 'no')")
    if do_background == 'yes':
        background_hex = input("What color do you want the background to be? (hex code)")
        paste_background = True

def process():
    open_dir = path(input_directory_path).glob('*.png') #Open input PNG directory, take only .pngs
    for img_path in open_dir.iterdir():
        # Process through REMGB
        img = pillow.open(img_path)
        output = remove(img, session=session)
        output.save(f"{transparent_directory_path}/{str(img_path)[:-4]}.png")


        #Add Background colour
        img = pillow.open(img_path).convert("RGBA")
        new_img_data = []
        bgColor = pillow.ImageColor.getrgb("#000338")
        transparent = (0,0,0,0)

        for data in img:
            if data[:3] == transparent:
                new_img_data.append(bgColor)
            else:
                new_img_data.append(data)
        img.putdata(new_img_data)
        img.save(f"/background_processed/{img_path}", "PNG")
        print(f"Processed {img_path}")
    print("Done")



if __name__ == '__main__':
    main()