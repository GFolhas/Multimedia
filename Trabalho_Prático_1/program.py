from contextlib import redirect_stderr
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np


"""
def encoder(image, YCbCr, compressionRate):
    return encodedImg

def decoder(image, conversionParams):
    return decodedImg
"""

def qualityChange(images, value):
    image = Image.open(images + ".bmp")
    img1High = image.convert("RGB")
    img1High.save(images + "_" + str(value) + ".jpg", quality=value)


def colorMapping(color1, color2):

    # Grey Scale
    repeats = 5;
    sample_number = 50; # isto é o valor default
    linGray = np.linspace(0., 1., sample_number).reshape(1, sample_number)
    linGray = np.repeat(linGray, repeats, axis=1).reshape(sample_number, repeats).T
    linGrayImg = np.zeros((repeats, sample_number, 3))

    # RGB
    linGrayImg[:, :, 0] = linGray # red
    linGrayImg[:, :, 1] = linGray # green
    linGrayImg[:, :, 2] = linGray # blue

    plt.figure()
    plt.title("Original Grey Scale")
    plt.imshow(linGrayImg)
    plt.axis('off')
    plt.show()

    colorlist=[color1, color2]
    colorMap = clr.LinearSegmentedColormap.from_list('cmap', colorlist, N=256) # gera um colormap linear suave (name, array de cores, niveis de quantizaçao rgb)
    plt.figure()
    plt.title(color1.capitalize() + "-" + color2.capitalize() + " Scale")
    plt.imshow(linGrayImg[:, :, 0], colorMap) #o parâmetro colormap só funciona se a imagem não for RGB
    plt.axis('off')
    plt.show()

    return colorMap


def imageColorMapping(images, colorMap, color1, color2):
    image = plt.imread(images + ".bmp")
    #redMap = clr.LinearSegmentedColormap.from_list('cmap', [(0, 0, 0), (1, 0, 0)], N=256)
    plt.figure()
    plt.title("Remapped Image (" + color1.capitalize() + "-" + color2.capitalize() + ")")
    plt.imshow(image[:, :, 0], colorMap)#o parâmetro colormap só funciona se a imagem não for RGB
    plt.axis('off')
    plt.show()


def rgbColorMapping(images):
    colors = ["red", "green", "blue"]
    vals = [(1,0,0), (0,1,0), (0,0,1)]
    image = plt.imread(images + ".bmp")
    data = []
    paths = []

    for i in range(len(colors)):
        map = clr.LinearSegmentedColormap.from_list('cmap', [(0, 0, 0), vals[i]], N=256)
        data.append(image[:,:,i])
        path = "imagens/RGB/" + images.split("/")[1] + colors[i].capitalize() + ".png"
        paths.append(path)
        plt.figure()
        #plt.title("Remapped Image (" + colors[i].capitalize() + ")")
        plt.imshow(image[:, :, i], map) # o parâmetro colormap só funciona se a imagem não for RGB
        plt.axis('off')
        plt.savefig("./" + path, bbox_inches='tight', pad_inches = 0)
        plt.show()
    
    return data, paths
    



def reverseRGBColorMapping(paths, data):
    colors = ["red", "green", "blue"]
    vals = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    array = []

    print(paths[0])

    for i in range(len(paths)):
        image = plt.imread(paths[i])

        val = image[:,:,i]
        array.append(val)
        #r = image[:,:,0] # lista valores RED para cada pixel 
        #g = image[:,:,1] # lista valores GREEN para cada pixel 
        #b = image[:,:,2] # lista valores BLUE para cada pixel 

        #print(int(image[:,:,0][0][0] * 256)) # 1º pixel blue -> 231

    # saiu do ciclo -> array tem a info de todos os pixeis, falta agora mudar a cor de cada um para uma combinaçao dos 3

    

def main():

    plt.close('all')

    ''' exercise 1 '''

    #qualityChange("imagens/logo", 100)


    ''' exercise 2 '''

    #encoder()
    #decoder()


    ''' exercise 3 '''

    #colors = ["purple", "gold"]
    #cm = colorMapping(colors[0], colors[1])
    #imageColorMapping("imagens/barn_mountains", cm, colors[0], colors[1])
    info, paths = rgbColorMapping("imagens/barn_mountains")
    reverseRGBColorMapping(paths, info)  # needs to be done
    """path = "imagens/RGB/barn_mountainsRed.png"
    image = plt.imread(path)
    print("FOUND AT " + path)
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.show()"""






if __name__ == "__main__":
    main()
