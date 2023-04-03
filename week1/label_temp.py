import os
import glob

# 0->0 / 2->1 / 4->2 / 6->3
label_cvt = {'0':'0', '2':'1', '4':'2', '6':'3'}
label_cvt_list = ['0', '2', '4', '6']

root_path = 'A:\\0cw_new_belt\\dataset\\'

label_paths = glob.glob(os.path.join(root_path, '*', 'images', '*'))

cnt = 0
for label_path in label_paths :
    image_path = label_path.replace('\\images\\', '\\labels\\')[:-4] + '.txt'
    if not (os.path.isfile(image_path)) :
        print(image_path)
        break

    if not (os.path.isfile(label_path)) :
        print(label_path)
        break



    # with open(label_path, 'r', encoding='utf-8') as f :
    #     lines = f.read()
    #
    # if lines == '' :
    #     os.remove(label_path)
    #     # label_path = label_path.replace('\\labels\\', '\\images\\')[:-4] + '.jpg'
    #     # os.remove(label_path)
    #
    # # with open(label_path, 'w', encoding='utf-8') as f :
    # #     for i in range(len(lines)) :
    # #         if lines[i] == '':
    # #             continue
    # #
    # #         if lines[i][0] in label_cvt_list :
    # #             lines[i] = label_cvt[lines[i][0]] + lines[i][1:]
    # #             f.write(lines[i] + '\n')
