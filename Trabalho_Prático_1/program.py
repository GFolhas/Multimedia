from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np


Tc = np.array([[0.299, 0.587, 0.114],[-0.168736, -0.331264, 0.5],[0.5, -0.418688, -0.081312]])
TcInverted = np.linalg.inv(Tc)


def encoder():
    pass

def decoder():
    pass


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


def joinRGB(x, y, c, array):
    
    vector = np.zeros((x,y,c))
    vector[:,:,0] = array[0]
    vector[:,:,1] = array[1]
    vector[:,:,2] = array[2]    
 
    vector = vector.astype(np.uint8)
    plt.figure()
    plt.title("RGB Components Added")
    plt.imshow(vector)
    plt.axis('off')
    plt.show()


def separateRGB(images):
    colors = ["red", "green", "blue"]
    vals = [(1,0,0), (0,1,0), (0,0,1)]
    image = plt.imread(images + ".bmp")

    #print(image)
    red = image[:,:,0]
    green = image[:,:,1]
    blue = image[:,:,2]

    x, y, c = image.shape
    #print(image.shape) -> (384, 512, 3)

    array = [red, green, blue]
    paths = []

    for i in range(len(colors)):
        map = clr.LinearSegmentedColormap.from_list('cmap', [(0, 0, 0), vals[i]], N=256)
        path = "imagens/RGB/" + images.split("/")[1] + colors[i].capitalize() + ".png"
        paths.append(path)
        plt.figure()
        plt.title("Remapped Image (" + colors[i].capitalize() + ")")
        plt.imshow(array[i], map) # o parâmetro colormap só funciona se a imagem não for RGB
        plt.axis('off')
        plt.savefig("./" + path, bbox_inches='tight', pad_inches = 0)
        plt.show()
    

    joinRGB(x, y, c, array)
    
  


def padding(image):
    img = plt.imread(image)
    
    p1 = img[:, :, 0]
    p2 = img[:, :, 1]
    p3 = img[:, :, 2]
    
    r, c = p1.shape
    org_r, org_c = p1.shape
    while (r%16) != 0:
        p1 = np.vstack([p1, p1[-1, :]])
        p2 = np.vstack([p2, p2[-1, :]])
        p3 = np.vstack([p3, p3[-1, :]])
        r, c = p1.shape
    
    while (c%16) != 0:
        p1 = np.hstack([p1, p1[:, -1]])
        p2 = np.hstack([p2, p2[:, -1]])
        p3 = np.hstack([p2, p2[:, -1]])
        r, c = p1.shape
    
    paddedImg = np.zeros((r, c, 3))
    paddedImg[:, :, 0] = p1
    paddedImg[:, :, 1] = p2
    paddedImg[:, :, 2] = p3
    #print("dim = ", paddedImg.shape)
    paddedImg = paddedImg.astype(np.uint8)
    plt.figure()
    plt.title("padded")
    plt.imshow(paddedImg)
    plt.axis('off')
    plt.show()
    unpad(paddedImg, org_r, org_c)
    
    
def unpad(paddedImg, r, c):
    unpaddedImg = paddedImg[:r, :c, :]
    #.print("dim = ", unpaddedImg.shape)
    unpaddedImg = unpaddedImg.astype(np.uint8)
    plt.figure()
    plt.title("unpadded")
    plt.imshow(unpaddedImg)
    plt.axis('off')
    plt.show()



def RGBYtoYCrCb(imgarray):
    image = plt.imread(imgarray + ".bmp")
    ycbcr = np.dot(image, Tc)
    ycbcr[:,:,[1,2]] += 128

    for i in range(3):
        plt.figure()
        plt.title("YCbCr from RBG Image (channel " + str(i) + ")")
        plt.imshow(ycbcr[:,:,i], "gray")
        plt.show()

    inv = ycbcr
    inv[:,:,[1,2]] -= 128
    rgb = np.dot(inv, TcInverted)

    rgb = rgb.round()
    rgb[rgb > 255] = 255
    rgb[rgb < 0] = 0

    rgb = rgb.astype(np.uint8)
    plt.figure()
    plt.title("RGB from YCbCr Image")
    plt.imshow(rgb)
    plt.show()


    


def main():

    plt.close('all')

    ''' exercise 1 '''

    #qualityChange("imagens/logo", 50)


    ''' exercise 2 '''

    encoder() # apenas tem um pass
    decoder() # apenas tem um pass


    ''' exercise 3 '''

    '''3.1 & 3.2'''
    colors = ["purple", "gold"]
    cm = colorMapping(colors[0], colors[1])

    '''3.3'''
    imageColorMapping("imagens/barn_mountains", cm, colors[0], colors[1])


    '''3.4'''

    separateRGB("imagens/peppers")


    ''' exercise 4 '''

    padding("imagens/barn_mountains.bmp")


    ''' exercise 5 '''

    RGBYtoYCrCb("imagens/barn_mountains")



if __name__ == "__main__":
    main()



'''
colormap references: https://matplotlib.org/1.2.1/mpl_examples/pylab_examples/show_colormaps.pdf
'''