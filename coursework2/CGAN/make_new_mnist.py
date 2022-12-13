import os
from PIL import Image
from array import *
from random import shuffle
import sys

def convert(ratio):
    # Load from and save to
    Names = [[['./GAN/synthesized_images/training-images/','syn'],['./data/png_MNIST/training-images','ori'],'train'],
             [['./GAN/synthesized_images/test-images/','syn'],['./data/png_MNIST/test-images','ori'],'t10k']]
    
    for name in Names:
        if name[2] == 'train':
            n_syn = 60000 * (ratio/100)
            n_ori = 60000 - n_syn
    
        if name[2] == 't10k':
            n_syn = 10000 * (ratio/100)
            n_ori = 10000 - n_syn
        
        data_image = array('B')
        data_label = array('B')

        FileList = []
        for x in name[:2]:
            if x[1] == 'ori' : n = n_ori
            else : n = n_syn
            if n==0: continue
            
            for dirname in os.listdir(x[0]): 
                path = os.path.join(x[0],dirname)
                cnt = 0
                DIRs = os.listdir(path)
                shuffle(DIRs)
                for filename in DIRs:
                    if filename.endswith(".png"):
                        FileList.append(os.path.join(x[0],dirname,filename))
                        cnt += 1
                        if cnt == n//10 : break
                        

        shuffle(FileList) # Usefull for further segmenting the validation set

        for filename in FileList:

            label = int(filename.split('/')[4])
            
            Im = Image.open(filename)

            pixel = Im.load()
            
            width, height = Im.size

            for x in range(0,width):
                for y in range(0,height):
                    if type(pixel[y,x]) == type((0,0,0)):
                        data_image.append(pixel[y,x][0]) #[0]: black-white image have same rgb val
                    else : data_image.append(pixel[y,x])

            data_label.append(label) # labels start (one unsigned byte each)

        hexval = "{0:#0{1}x}".format(len(FileList),6) # number of files in HEX

        # header for label array

        header = array('B')
        header.extend([0,0,8,1,0,0])
        header.append(int('0x'+hexval[2:][:2],16))
        header.append(int('0x'+hexval[2:][2:],16))
        
        data_label = header + data_label

        # additional header for images array
        
        if max([width,height]) <= 256:
            header.extend([0,0,0,width,0,0,0,height])
        else:
            raise ValueError('Image exceeds maximum size: 256x256 pixels')

        header[3] = 3 # Changing MSB for image data (0x00000803)
        
        data_image = header + data_image

        output_file = open(f'data/zipped_synthesized_images/{ratio}/MNIST/raw/'+name[2]+'-images-idx3-ubyte', 'wb')
        data_image.tofile(output_file)
        output_file.close()

        output_file = open(f'data/zipped_synthesized_images/{ratio}/MNIST/raw/'+name[2]+'-labels-idx1-ubyte', 'wb')
        data_label.tofile(output_file)
        output_file.close()
        
        os.system(f'gzip data/zipped_synthesized_images/{ratio}/MNIST/raw/'+name[2]+'-images-idx3-ubyte')
        os.system(f'gzip data/zipped_synthesized_images/{ratio}/MNIST/raw/'+name[2]+'-labels-idx1-ubyte')

        output_file = open(f'data/zipped_synthesized_images/{ratio}/MNIST/raw/'+name[2]+'-images-idx3-ubyte', 'wb')
        data_image.tofile(output_file)
        output_file.close()

        output_file = open(f'data/zipped_synthesized_images/{ratio}/MNIST/raw/'+name[2]+'-labels-idx1-ubyte', 'wb')
        data_label.tofile(output_file)
        output_file.close()
        # gzip resulting files


if __name__ == "__main__":
    
    convert(10)  #10%
    convert(20) #20%
    convert(50) #50%
    convert(100) #100%
        