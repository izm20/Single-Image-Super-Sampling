import os
dir_name = './'
dir = os.listdir(dir_name)
number = 100000
for ifile in dir:
     new_name = ' ' + dir_name + 'frame' + str(number) + '.jpg'
     command = 'mv ' + dir_name + ifile + new_name
     os.system(command)
     number = number + 1
     print('Saved the image: ' + new_name)
