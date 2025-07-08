######
# Converter from RealityCapture csv file format for tags to Metashape fileformat
#
# Caution : This python code is not optimized and ressembles as a draft, however it works as intended.
######
# Modify the variable: tags_file
# For example:
# tags_file = "C:/projectN/AprilTags_36h11_on_images/20250422_1234-36h11_relative-1,0.25,2.0,0.0_0.785,10.0,10,300,5.txt"
# Choose the relative version, not the absolute one
# Then execute this script into Metashape
######

""" https://www.agisoft.com/forum/index.php?topic=3008.0
https://www.agisoft.com/pdf/metashape_python_api_2_0_0.pdf
"""

#import Metashape
import csv


def parser_csv():
    tags_file = "C:/projectN/AprilTags_36h11_on_images/20250422_1234-36h11_relative-1,0.25,2.0,0.0_0.785,10.0,10,300,5.txt"
    with open(tags_file, 'r') as ref_file:
        csv_reader = csv.reader(ref_file)
        array_ref = list(csv_reader)
        
        dico_by_tag = {}
        dico_by_img = {}
        
        #print(array_ref[0])
        for i, l in enumerate(array_ref):
            img = l[0].strip()
            tag = l[1].strip()
            x2D = l[2].strip()
            y2D = l[3].strip()
            
            if tag not in dico_by_tag:
                dico_by_tag[tag] = [(img, x2D, y2D)]
            else:
                dico_by_tag[tag].append((img, x2D, y2D))
            
            if img not in dico_by_img:
                dico_by_img[img] = [(tag, x2D, y2D)]
            else:
                dico_by_img[img].append((tag, x2D, y2D))
                
        #print(dico_by_tag['161'][0])
    return dico_by_tag, dico_by_img
        
        #list_all_projections = []
def makeMakers(chunk, dico_by_tag):
        dico_imgs_cam = {}
        """for cam in chunk.cameras:
            dico_imgs_cam[cam.label] = cam.key
            #print(cam.label)
            #print(cam.key)"""

        for i, tag in enumerate(dico_by_tag):
            print(i)
            print(tag)
            #list_projections = []
            marker = chunk.addMarker()
            marker.label = tag
            
            for triplet in dico_by_tag[tag]:
                img, x2D, y2D = triplet
                label = img[:-4] # Crop extension
                x2D = float(x2D)
                y2D = float(y2D)
                #target = Metashape.Target(i, coords2D, radius=0)
                #list_all_projections.append((img, i))
                    #print(cam.label)
                coords2D = Metashape.Vector([x2D,y2D])
                #list_projections.append((dico_imgs_cam[label],coords2D))
                for cam in chunk.cameras:
                    if cam.label == img[:-4]:
                        marker.projections[cam] = Metashape.Marker.Projection(coords2D, True)
            #marker.projections = list_projections
            #coords2D = Metashape.Vector([x2D,y2D])
            #target = Metashape.Target(i, coords2D, radius=0) # i here has to be integer, so not the "tag name" as in "tag variable"
            #projections = ((camera, target),(camera, target),...)
            #projections = (
        #print(list_all_projections)
        #createMarkers(chunk, projections)
        

            """# assign the marker's x,y coords in the current cam
                        marker.projections[cam] = pixel_coord
                        # Add the world coordinates to the marker
                        marker.Reference.location = MS.Vector([cx, cy, cz])
                        # Set the accuracy of the marker's coords
                        marker.Reference.accuracy = MS.Vector(accuracy)
                        # enable the marker
                        marker.Reference.enabled = True"""


"""chunk = Metashape.app.document.chunk

images = chunk.cameras"""

# List of indexes and names (without .jpg extension)
#for im in chunk.cameras: print(str(im.key)+" "+str(im.label))

dico_by_tag, dico_by_img = parser_csv()
chunk = Metashape.app.document.chunk
makeMakers(chunk,dico_by_tag)