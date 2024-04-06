import json

def openTask1():
    with open('face_landmarker_v2_with_blendshapes.task', 'rb') as file: #'r'
        tasks = json.load(file)
        #tasks = file.read()
        #for task in tasks:
        #    print(task)
        print(tasks)
