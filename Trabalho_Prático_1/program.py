from PIL import Image
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

    # rgb
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

    for i in range(len(colors)):
        map = clr.LinearSegmentedColormap.from_list('cmap', [(0, 0, 0), vals[i]], N=256)
        plt.figure()
        plt.title("Remapped Image (" + colors[i].capitalize() + ")")
        plt.imshow(image[:, :, 0], map)#o parâmetro colormap só funciona se a imagem não for RGB
        plt.axis('off')
        plt.savefig("./imagens/RGB/" + images.split("/")[1] + colors[i].capitalize() + ".png")
        plt.show()


def reverseRGBColorMapping(images):
    """
    # Honestamente nao faço puto de ideia, nao encontro nada na net e n estou bem a ver como fzr isto a n ser criar um novo colormap à mão atraves da soma dos valores hex de cada mapa anterior

    colors = ["red", "green", "blue"]
    vals = [(1,0,0), (0,1,0), (0,0,1)]
    image = plt.imread(images + ".bmp")
    colormaps = []

    for i in range(len(colors)):
        map = clr.LinearSegmentedColormap.from_list('cmap', [(0, 0, 0), vals[i]], N=85)
        colormaps.append(map)

    c = np.vstack((colormaps[0], colormaps[1], colormaps[2]))
    print(c)
    print("POG")
    #map = clr.LinearSegmentedColormap.from_list('cmap', c, N = 256)
    plt.figure()
    #plt.title("Remapped Image (" + colors[i].capitalize() + ")")
    plt.imshow(image[:, :, 0], c)#o parâmetro colormap só funciona se a imagem não for RGB
    plt.axis('off')
    #plt.savefig("./imagens/RGB/" + images.split("/")[1] + colors[i].capitalize() + ".png")
    plt.show()
    """

def main():

    plt.close('all')

    ''' exercise 1 '''

    #qualityChange("imagens/logo", 100)


    ''' exercise 2 '''

    #encoder()
    #decoder()


    ''' exercise 3 '''

    colors = ["purple", "gold"]
    cm = colorMapping(colors[0], colors[1])
    imageColorMapping("imagens/barn_mountains", cm, colors[0], colors[1])
    rgbColorMapping("imagens/barn_mountains")
    #reverseRGBColorMapping("imagens/barn_mountains")  # needs to be done




if __name__ == "__main__":
    main()
