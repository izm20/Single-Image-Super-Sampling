from PIL import Image
import os
import sys


if len(sys.argv) == 4:
    dir = os.listdir(sys.argv[1])
    width = int(sys.argv[2])
    height = int(sys.argv[3])
    dirname = "./custom_resolution_images/"
    for ifile in dir:
        myImage = Image.open(sys.argv[1] + ifile)
        imResult = myImage.resize((width, height), Image.BICUBIC)
        imResult.save(dirname + ifile)
        print('Saved the image ' + ifile)
else:
    print("Try: python3 toCustomResolution.py dirWithPhotos/ $width $height")