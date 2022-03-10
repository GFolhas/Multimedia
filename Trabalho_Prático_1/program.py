
from PIL import Image
from cv2 import CAP_PROP_XI_COUNTER_VALUE
from scipy import fftpack as fft
from scipy import ifft
from matplotlib import cm

import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import cv2

Tc = np.array([[0.299, 0.587, 0.114],[-0.168736, -0.331264, 0.5],[0.5, -0.418688, -0.081312]])
TcInverted = np.linalg.inv(Tc)


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
    

    return x, y, c, array

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

    return

def padding(image):
    img = plt.imread(image + ".bmp")
    
    p1 = img[:, :, 0]
    p2 = img[:, :, 1]
    p3 = img[:, :, 2]
    
    
    r, c = p1.shape
    countC = c%16
    countR = r%16
    org_r, org_c = p1.shape
    if (countR) != 0:
        p1 = np.vstack((p1, np.tile(p1[-1, :], (countR, 1))))
        p2 = np.vstack((p2, np.tile(p2[-1, :], (countR, 1))))
        p3 = np.vstack((p3, np.tile(p3[-1, :], (countR, 1))))
        r, c = p1.shape
    
    if (countC) != 0:
        p1 = np.hstack((p1, np.tile(p1[:, -1], (countC, 1).T)))
        p2 = np.hstack((p2, np.tile(p2[:, -1], (countC, 1).T)))
        p3 = np.hstack((p3, np.tile(p3[:, -1], (countC, 1).T)))
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
    
    return paddedImg, org_r, org_c
    
    
def unpad(paddedImg, org_r, org_c):
    unpaddedImg = paddedImg[:org_r, :org_c, :]
    #.print("dim = ", unpaddedImg.shape)
    unpaddedImg = unpaddedImg.astype(np.uint8)
    plt.figure()
    plt.title("unpadded")
    plt.imshow(unpaddedImg)
    plt.axis('off')
    plt.show()



def RGBtoYCrCb(image):
    imgarray = plt.imread(image + ".bmp")
    ycbcr = np.dot(imgarray, Tc)
    ycbcr[:,:,[1,2]] += 128
    
    y = ycbcr[:,:,0]
    Cb = ycbcr[:,:,1]
    Cr = ycbcr[:,:,2]

    for i in range(3):
        plt.figure()
        plt.title("YCbCr from RBG Image (channel " + str(i) + ")")
        plt.imshow(ycbcr[:,:,i], "gray")
        plt.show()

    return y, Cb, Cr, ycbcr


def YCbCrtoRGB(ycbcr):
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
    
    return rgb

def subsampling(Cb, Cr, ratio, inter):
    cbRatio = ratio[1]/ratio[0]

    if ratio[2] == 0:
        if ratio[1] == 4:
            crRatio = 0.5
        else:
            crRatio = cbRatio
    else:
        crRatio = 1
    
    cbStep = int(1//cbRatio)
    crStep = int(1//crRatio)

    if inter:
        dsCbInterp = cv2.resize(Cb, None, fx=cbRatio, fy=crRatio, interpolation=cv2.INTER_LINEAR)
        dsCrInterp = cv2.resize(Cr, None, fx=cbRatio, fy=crRatio, interpolation=cv2.INTER_LINEAR)

        plt.subplots_adjust(left=0.01, right=0.99, wspace=0.1)
        plt.subplot(1, 2, 1)
        plt.title("Cb Downsampled with Interpolation")
        plt.imshow(dsCbInterp, "gray")
        plt.subplot(1, 2, 2)
        plt.title("Cr Downsampled with Interpolation")
        plt.imshow(dsCrInterp, "gray")
        plt.show()

        return  cbStep, crStep, dsCbInterp, dsCrInterp

    else:
        cbDown = Cb[::crStep, ::crStep]
        crDown = Cr[::cbStep, ::cbStep]

        plt.subplots_adjust(left=0.01, right=0.99, wspace=0.1)
        plt.subplot(1, 2, 1)
        plt.title("Cb Downsampled without Interpolation")
        plt.imshow(cbDown, "gray")
        plt.subplot(1, 2, 2)
        plt.title("Cr Downsampled without Interpolation")
        plt.imshow(crDown, "gray")
        plt.show()

        return  cbStep, crStep, cbDown, crDown

    

def upsampler(cbStep, crStep, dsCb, dsCr, inter):

    usCb = np.repeat(dsCb, cbStep, axis=1)
    usCb = np.repeat(usCb, crStep, axis=0)

    usCr = np.repeat(dsCr, cbStep, axis=1)
    usCr = np.repeat(usCr, crStep, axis=0)

    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.1)
    plt.subplot(1, 2, 1)
    plt.title("Cb Upsampled with Repetition")
    plt.imshow(usCb, "gray")
    plt.subplot(1, 2, 2)
    plt.title("Cr Upsampled with Repetition")
    plt.imshow(usCr, "gray")
    plt.show()

    usCb = cv2.resize(dsCb, None, fx=cbStep, fy=crStep, interpolation=cv2.INTER_LINEAR)
    usCr = cv2.resize(dsCr, None, fx=cbStep, fy=crStep, interpolation=cv2.INTER_LINEAR)

    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.1)
    plt.subplot(1, 2, 1)
    plt.title("Cb Upsampled with Interpolation")
    plt.imshow(usCb, "gray")
    plt.subplot(1, 2, 2)
    plt.title("Cr Upsampled with Interpolation")
    plt.imshow(usCr, "gray")
    plt.show()

    print()
    print("Downsampling Cb size = ", dsCb.shape)
    print("Upsampling Cb size with Repetition = ", usCb.shape)
    print("Upsampling Cb size with Interpolation = ", usCb.shape)
    print()
    print("Downsampling Cr size = ", dsCr.shape)
    print("Upsampling Cr size with Repetition = ", usCr.shape)
    print("Upsampling Cr size with Interpolation = ", usCr.shape)
    print()


def dct2(images):

    img = plt.imread(images + ".bmp")
    cm_grey = clr.LinearSegmentedColormap.from_list('greyMap', [(0, 0, 0), (1, 1, 1)], 256)

    dctImg = fft.dct(fft.dct(img, norm="ortho").T, norm="ortho").T
    dctLogImg = np.log(np.abs(dctImg) + 0.0001)

    fig = plt.figure(figsize=(20, 20))

    fig.add_subplot(1, 3, 1)
    plt.imshow(img, cm_grey)
    plt.title('original')
    plt.axis('image')

    fig.add_subplot(1, 3, 2)
    plt.imshow(dctImg, cm_grey)
    plt.title('DCT')
    plt.axis('image')

    fig.add_subplot(1, 3, 3)
    plt.imshow(dctLogImg, cm_grey)
    plt.title('DCT log')
    plt.axis('image')
    plt.show()

    invDctImg = fft.idct(fft.idct(dctImg, norm="ortho").T, norm="ortho").T

    fig = plt.figure(figsize=(20, 20))

    fig.add_subplot(1, 4, 1)
    plt.imshow(img, cm_grey)
    plt.title('original')
    plt.axis('image')

    fig.add_subplot(1, 4, 2)
    plt.imshow(invDctImg, cm_grey)
    plt.title('IDCT')
    plt.axis('image')
    plt.show()

    fig = plt.figure(figsize=(5, 5))
    diffImg = img-invDctImg
    diffImg[diffImg < 0.000001] = 0.

    plt.imshow(diffImg, cm_grey)
    plt.title('original - invDCT')
    plt.axis('image')
    plt.show()

def idct(X: np.ndarray) -> np.ndarray:
    return fft.idct(fft.idct(X, norm="ortho").T, norm="ortho").T

def dct(X: np.ndarray) -> np.ndarray:
    return fft.dct(fft.dct(X, norm="ortho").T, norm="ortho").T

def blockIdct(x: np.ndarray, size):
    h, w = x.shape
    newImg = np.zeros(x.shape)
    for i in range(0, h, size):
        for j in range(0, w, size):
            newImg[i:i+size, j:j+size] = idct(x[i:i+size, j:j+size])
    return newImg 

def blockDct(x: np.ndarray, size):
    h, w = x.shape
    newImg = np.zeros(x.shape)
    for i in range(0, h, size):
        for j in range(0, w, size):
            newImg[i:i+size, j:j+size] = dct(x[i:i+size, j:j+size])
    return newImg   
    
def exer(y, cb, cr):
    block = 8
    #y, Cb, Cr = RGBtoYCrCb("imagens/barn_mountains")
    #x, y1, c = plt.imread("imagens/barn_mountains.bmp").shape
    
    y = blockDct(y, size=block)
    cb = blockDct(cb, size=block)
    cr = blockDct(cr, size=block)
    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.1)
    plt.title("DCT 8x8")
    plt.subplot(1, 3, 1)
    plt.imshow(y, "gray")
    plt.subplot(1, 3, 2)
    plt.imshow(cb, "gray")
    plt.subplot(1, 3, 3)
    plt.imshow(cr, "gray")
    plt.show()

    y = blockIdct(y, size=block)
    cb = blockIdct(cb, size=block)
    cr = blockIdct(cr, size=block)
    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.1)
    plt.title("Inverted DCT 8x8")
    plt.subplot(1, 3, 1)
    plt.imshow(y, "gray")
    plt.subplot(1, 3, 2)
    plt.imshow(cb, "gray")
    plt.subplot(1, 3, 3)
    plt.imshow(cr, "gray")
    plt.show()

    block = 64
    y = blockDct(y, size=block)
    cb = blockDct(cb, size=block)
    cr = blockDct(cr, size=block)
    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.1)
    plt.title("DCT 32x32")
    plt.subplot(1, 3, 1)
    plt.imshow(y, "gray")
    plt.subplot(1, 3, 2)
    plt.imshow(cb, "gray")
    plt.subplot(1, 3, 3)
    plt.imshow(cr, "gray")
    plt.show()

    y = blockIdct(y, size=block)
    cb = blockIdct(cb, size=block)
    cr = blockIdct(cr, size=block)
    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.1)
    plt.title("Inverted DCT 8x8")
    plt.subplot(1, 3, 1)
    plt.imshow(y, "gray")
    plt.subplot(1, 3, 2)
    plt.imshow(cb, "gray")
    plt.subplot(1, 3, 3)
    plt.imshow(cr, "gray")
    plt.show()
    
    #𝐈𝐭'𝐬 𝐚 𝐰𝐢𝐥𝐝 𝐩𝐨𝐩𝐩𝐥𝐢𝐞𝐫
def encoder(image):
    
    x, y1, c, array = separateRGB(image)
    padded_img, org_r, org_c = padding(image)
    y , cb, cr, ycbcr = RGBtoYCrCb(image)
    cbStep, crStep, SubCb, SubCr = subsampling(cb, cr, (4,2,0), True)

    dct2(image)
    
    
    return x, y1, c, array, padded_img, org_r, org_c, y, cb, cr, ycbcr, cbStep, crStep, SubCb, SubCr        
    
        
def decoder(x, y1, c, array, padded_img, org_r, org_c, y, cb, cr, ycbcr, cbStep, crStep, SubCb, SubCr):
    
    joinRGB(x, y1, c, array)
    unpad(padded_img, org_r, org_c)
    YCbCrtoRGB(ycbcr)
    upsampler(cbStep, crStep, SubCb, SubCr, True)
    exer(y, cb, cr)
    return


def main():

    plt.close('all')

    image = "imagens/peppers"
    inter = True
    x, y1, c, array, padded_img, org_r, org_c, y, cb, cr, ycbcr, cbStep, crStep, SubCb, SubCr = encoder(image)
    decoder(x, y1, c, array, padded_img, org_r, org_c, y, cb, cr, ycbcr, cbStep, crStep, SubCb, SubCr)  


if __name__ == "__main__":
    main()



'''
colormap references: https://matplotlib.org/1.2.1/mpl_examples/pylab_examples/show_colormaps.pdf
'''