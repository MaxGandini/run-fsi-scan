from pathlib import Path

FROM_IMAGE = 0
TO_IMAGE = 50

NONE_STANDARD = 5
#SCANNER   

MAX_DEPTH= 10
STRICTNESS = 0.15
TOP_INTENSITY = 210
BOTTOM_INTENSITY = 15

#FILTER
ADAPTIVE_VALUE = 37
KSIZE=3 #tamanio del kernel (valor global)

#Morph params
CHARACTERISTIC_LENGTH = 5
CHARACTERISTIC_THRESH = CHARACTERISTIC_LENGTH*4
EPSILON = CHARACTERISTIC_LENGTH + 1 


cwd = Path.cwd()
PROJECT_FOLDER = cwd.parent.parent / "Slices_13"
OUTPUT_FOLDER = cwd  
TEST_FOLDER = Path("test_folder")
PROJECTION_FOLDER = TEST_FOLDER / "projection_folder"
PROJ_ON_IMAGE_FOLDER = TEST_FOLDER / "projection_folder" / "projections_on_image"
CROSSINGS_FOLDER = TEST_FOLDER / "projection_folder" / "crossings"
