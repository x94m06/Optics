import os

image_path = 'D:/GitHub/Optics/1'
img_name = next(os.walk(image_path))[2]
#print(type(img_name),'\n',img_name,'\n')
print(img_name)