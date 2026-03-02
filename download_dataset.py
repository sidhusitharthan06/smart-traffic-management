from roboflow import Roboflow
import os

print("Step 1: Connecting to Roboflow...")
rf = Roboflow(api_key="3gBvogdDjkGuKZn2p7It")

print("Step 2: Loading project...")
project = rf.workspace("cats-vs-dogs-detection").project("vehicle-detection-whqwa")

print("Step 3: Getting version 2...")
version = project.version(2)

print("Step 4: Downloading in YOLOv8 format...")
dataset = version.download("yolov8")

print("Step 5: Download complete!")
print("Dataset location:", dataset.location)

# Check what was downloaded
for root, dirs, files in os.walk(dataset.location):
    level = root.replace(dataset.location, '').count(os.sep)
    indent = ' ' * 2 * level
    print('{}{}/'.format(indent, os.path.basename(root)))
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:
        print('{}{}'.format(subindent, file))
    if len(files) > 5:
        print('{}... and {} more files'.format(subindent, len(files) - 5))
