######
# Created to generate masks with Laplacian Pyramid method on blurred areas for close-range Structure from Motion (SfM)
# Bonus: refine detection of AprilTags centers and write it in csv files
#
# Author: Yannick FAURE
# Licence: GPL v3
#
# Caution : This python code is not optimized and ressembles as a draft, however it works as intended.
######
# This module is mandatory to clean exif rotations before running masks computing, it is called by MAIN
######

import imageio # Can save EXIF without changing image quality
from exif import Image
from os import listdir
from os.path import isfile, join, basename

class Clean_some_exif(object):
    images_files = []
    
    def list_files(self, folder):
        self.folder = folder
        self.image_files = [f for f in listdir(self.folder) if isfile(join(self.folder, f)) and f.lower().endswith(('.jpg', '.JPG'))]


    def ifRotExif(self):
        orientation = False
        for im in self.image_files:
            #print(im)
            with open(join(self.folder, im), 'rb') as f:
                img = Image(f)
                #print(img.list_all())
                if 'orientation' in img.list_all():
                    print(str(im)+" "+str(img.orientation))
                    orientation = True
        if orientation == True:
            return True
        else:
            return False


    def clean_orientation_tag(self):
        for im in self.image_files:
            print(im)
            with open(join(self.folder, im), 'rb') as f:
                img = Image(f)
                print(img.list_all())
                if 'orientation' in img.list_all():
                    print(img.orientation)
                    del img.orientation
                    with open(join(self.folder, im), 'wb') as f_out:
                        f_out.write(img.get_file())
                        
    def clean_gps_tag(self):
        for im in self.image_files:
            print(im)
            with open(join(self.folder, im), 'rb') as f:
                img = Image(f)
                print(img.list_all())
                gps_tags = [tag for tag in img.list_all() if tag.startswith('gps_')]
                if gps_tags:
                    print(f"Coordonnées GPS trouvées : {gps_tags}")
                    for tag in gps_tags:
                        delattr(img, tag)
                    with open(join(self.folder, im), 'wb') as f_out:
                        f_out.write(img.get_file())
    
    def clean_rot(self,folder_path):
        self.folder = folder_path
        self.list_files()
        self.clean_orientation_tag()

    def clean_gps(self,folder_path):
        self.folder = folder_path
        self.list_files()
        self.clean_gps_tag()

"""clean = Clean_some_exif()
clean.list_files()
clean.clean_orientation_tag()"""