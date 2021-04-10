import glob
import csv

row_data = [
    ['left_eye_img', 'x_coord']
]


def get_id(file):
    if 'cursor' in file:
        return file.split('-')[1]
    else:
        return file.split('-')[0].split('/')[1]

def get_x_value_cursor(file):
    return int(file.split('-')[2])

data_temp = {}
for filepath in glob.iglob('tmp/*.*'):
    if 'head' not in filepath:
        if '-1.jpg' not in filepath:
            id = get_id(filepath)
            if 'cursor' in filepath:
                filepath = get_x_value_cursor(filepath)
            else:
                filepath = filepath.replace('tmp/', '')
            if id not in data_temp:
                data_temp[id] = [filepath]
            else:
                data_temp[id].append(filepath)

for x, y in data_temp.items():
    x_coord_value = y[0] if isinstance(y[0], int) else y[1]
    left_eye_img = y[1] if isinstance(y[0], int) else y[0]
    row_data.append([left_eye_img, x_coord_value])


with open('dataset.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(row_data)
