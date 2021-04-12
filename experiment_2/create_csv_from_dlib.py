import glob
import csv

row_data = [
    ['left_eye_img', 'x_coord']
]

def get_id(file):
    if 'cursor' in file:
        return file.split('-')[1]
    else:
        return file.split('-')[0]

def get_x_coord_from_file(file):
    return int(file.split('-')[2])

data_temp = {}
for filepath in glob.iglob('dlib_data/*.*'):
    filepath = filepath.replace('dlib_data/', '')
    id_ = get_id(filepath)
    if 'cursor' in filepath:
        filepath = get_x_coord_from_file(filepath)
    if id_ in data_temp:
        data_temp[id_].append(filepath)
    else:
        data_temp[id_] = [filepath]


for value in data_temp.values():
    x_coord_value = value[0] if isinstance(value[0], int) else value[1]
    left_eye_img = value[1] if isinstance(value[0], int) else value[0]
    row_data.append([left_eye_img, x_coord_value])

with open('dlib_dataset2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(row_data)
