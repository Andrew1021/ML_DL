import pandas as pd
import numpy as np
import os
import wget
import tarfile
from sklearn import *

url = 'http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz'
tgz_file = 'lfw_funneled.tgz'
images_folder = 'lfw_funneled'


def download_file(url):
    #
    if not os.path.isfile(tgz_file):
        print("Downloading file...\n")
        wget.download(url, tgz_file)
        # dateDownloaded = !date  # Calling Linux
        # print(dateDownloaded)
    return 0


def extract():
    try:
        print("Extracting...")
        tar = tarfile.open(tgz_file, "r:gz")
        tar.extractall()
        tar.close()
    except:
        print("something went wrong")


def find_images():
    list_of_persons = []
    if not os.path.isdir(images_folder):
        if not os.path.isfile(tgz_file):
            download_file(url)
        else:
            print("tgz-File exists")
            extract()
    else:
        for folder in os.listdir(images_folder):
            subfolder = os.path.join(images_folder, folder)
            if os.path.isdir(subfolder):
                count = len(os.listdir(subfolder))
                if count >= 70:
                    list_of_persons.append(folder)
                    print("{} with {} images".format(folder, count))


download_file(url)
find_images()
