from PIL import Image
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

# Exercise 3 - Separate and join RGB channels

# Separate RGB
def separateRGB(images, flag):
    colors = ["red", "green", "blue"]
    vals = [(1,0,0), (0,1,0), (0,0,1)]
    image = plt.imread(images + ".bmp")

    #print(image)
    red = image[:,:,0]
    green = image[:,:,1]
    blue = image[:,:,2]

    
    #print(image.shape) -> (384, 512, 3)

    array = [red, green, blue]
    paths = []

    for i in range(len(colors)):
        map = clr.LinearSegmentedColormap.from_list('cmap', [(0, 0, 0), vals[i]], N=256)
        path = "imagens/RGB/" + images.split("/")[1] + colors[i].capitalize() + ".png"
        paths.append(path)
        if flag:
            plt.figure()
            plt.title("Remapped Image (" + colors[i].capitalize() + ")")
            plt.imshow(array[i], map) # o parâmetro colormap só funciona se a imagem não for RGB
            plt.axis('off')
            plt.savefig("./" + path, bbox_inches='tight', pad_inches = 0)
            plt.show()
    

    return array[0], array[1], array[2]
    
# Join RGB
def joinRGB(r, g, b, flag):
    
    rgb = np.zeros((r,g,3))
    rgb[:,:,0] = r
    rgb[:,:,1] = g
    rgb[:,:,2] = b    
 
    rgb = rgb.astype(np.uint8)
    if flag:
        plt.figure()
        plt.title("RGB Components Added")
        plt.imshow(rgb)
        plt.axis('off')
        plt.show()

    return rgb

# Exercise 4 - Image Padding
  
# Padding  
def padding(r, g, b, flag):
    #img = plt.imread(image + ".bmp")    
    
    h, c = r.shape
    countH = h%16
    countC = c%16
    org_h, org_c = r.shape
    if (countH) != 0:
        r = np.vstack((r, np.tile(r[-1, :], (countH, 1))))
        g = np.vstack((g, np.tile(g[-1, :], (countH, 1))))
        b = np.vstack((b, np.tile(b[-1, :], (countH, 1))))
        h, c = r.shape
    
    if (countC) != 0:
        r = np.hstack((r, np.tile(r[:, -1], (countC, 1).T)))
        g = np.hstack((g, np.tile(g[:, -1], (countC, 1).T)))
        b = np.hstack((b, np.tile(b[:, -1], (countC, 1).T)))
        h, c = r.shape
    
    paddedImg = np.zeros((h, c, 3))
    paddedImg[:, :, 0] = r
    paddedImg[:, :, 1] = g
    paddedImg[:, :, 2] = b
    #print("dim = ", paddedImg.shape)
    paddedImg = paddedImg.astype(np.uint8)
    
    if flag:
        plt.figure()
        plt.title("padded")
        plt.imshow(paddedImg)
        plt.axis('off')
        plt.show()
    
    return r, g, b

# Unpadding    
def unpad(paddedImg, org_r, org_c, flag):
    unpaddedImg = paddedImg[:org_r, :org_c, :]
    #.print("dim = ", unpaddedImg.shape)
    unpaddedImg = unpaddedImg.astype(np.uint8)
    
    if flag:
        plt.figure()
        plt.title("unpadded")
        plt.imshow(unpaddedImg)
        plt.axis('off')
        plt.show()
    
    return unpaddedImg

# Exercise 5 - Conversion to YCbCr and reverse it

# RGB to YCbCR
def RGBtoYCrCb(image, flag):
    # Flag para fzr plots
    imgarray = plt.imread(image + ".bmp")
    ycbcr = np.dot(imgarray, Tc)
    ycbcr[:,:,[1,2]] += 128
    
    y = ycbcr[:,:,0]
    Cb = ycbcr[:,:,1]
    Cr = ycbcr[:,:,2]

    if flag:
        for i in range(3):
            plt.figure()
            plt.title("YCbCr from RBG Image (channel " + str(i) + ")")
            plt.imshow(ycbcr[:,:,i], "gray")
            plt.show()

    return y, Cb, Cr

# YCbCr to RGB
def YCbCrtoRGB(y: np.ndarray, cb: np.ndarray, cr: np.ndarray, flag):
    '''inv = np.zeros((y,cb,3))
    inv[:,:,0] = y
    inv[:,:,1] = cb
    inv[:,:,2] = cr
    
    print(inv)
    inv[:,:,[1,2]] -= 128
    rgb = np.dot(inv, TcInverted)
    '''
    inv = np.empty(shape, dtype=y.dtype)
    inv[:, :, 0] = y
    inv[:, :, 1] = cb
    inv[:, :, 2] = cr

    inv[:,:,[1,2]] -= 128

    rgb = rgb.round()
    rgb[rgb > 255] = 255
    rgb[rgb < 0] = 0

    rgb = rgb.astype(np.uint8)
    if flag:
        plt.figure()
        plt.title("RGB from YCbCr Image")
        plt.imshow(rgb)
        plt.show()
    
    return rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

# Exercise 6 - SubSampling of YCbCr channels

# SubSampling
def subsampling(Cb, Cr, ratio, inter, flag):
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

        if flag:
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
        if flag:
            plt.subplots_adjust(left=0.01, right=0.99, wspace=0.1)
            plt.subplot(1, 2, 1)
            plt.title("Cb Downsampled without Interpolation")
            plt.imshow(cbDown, "gray")
            plt.subplot(1, 2, 2)
            plt.title("Cr Downsampled without Interpolation")
            plt.imshow(crDown, "gray")
            plt.show()

        return  cbStep, crStep, cbDown, crDown

# UpSampling
def upsampler(cbStep, crStep, dsCb, dsCr, inter, flag):

    # Flag inter

    if inter:
        usCb = np.repeat(dsCb, cbStep, axis=1)
        usCb = np.repeat(usCb, crStep, axis=0)

        usCr = np.repeat(dsCr, cbStep, axis=1)
        usCr = np.repeat(usCr, crStep, axis=0)

        if flag:

            plt.subplots_adjust(left=0.01, right=0.99, wspace=0.1)
            plt.subplot(1, 2, 1)
            plt.title("Cb Upsampled with Repetition")
            plt.imshow(usCb, "gray")
            plt.subplot(1, 2, 2)
            plt.title("Cr Upsampled with Repetition")
            plt.imshow(usCr, "gray")
            plt.show()

    else: 
        usCb = cv2.resize(dsCb, None, fx=cbStep, fy=crStep, interpolation=cv2.INTER_LINEAR)
        usCr = cv2.resize(dsCr, None, fx=cbStep, fy=crStep, interpolation=cv2.INTER_LINEAR)

        if flag:
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

# Exercise 7 - 

# DCT
def dct2(y, cb, cr):



    dctImg = fft.dct(fft.dct(img, norm="ortho").T, norm="ortho").T
    dctLogImg = np.log(np.abs(dctImg) + 0.0001)

    fig = plt.figure(figsize=(20, 20))

    fig.add_subplot(1, 3, 1)
    plt.imshow(img, "gray")
    plt.title('original')
    plt.axis('image')

    fig.add_subplot(1, 3, 2)
    plt.imshow(dctImg, "gray")
    plt.title('DCT')
    plt.axis('image')                       

    fig.add_subplot(1, 3, 3)
    plt.imshow(dctLogImg, "gray")
    plt.title('DCT log')
    plt.axis('image')
    plt.show()

    return dctImg[:,:,0], dctImg[:,:,1], dctImg[:,:,2], #dctImg.shape

# IDCT
def idct2(ydct, cbdct, crdct, shape):

    dctImg = np.zeros(shape)
    dctImg[:,:,0] = ydct
    dctImg[:,:,1] = cbdct
    dctImg[:,:,2] = crdct

    # abaixo decoder
    invDctImg = fft.idct(fft.idct(dctImg, norm="ortho").T, norm="ortho").T
    invs = invDctImg
    invDctImg = np.log(np.abs(dctImg) + 0.0001)

    fig = plt.figure(figsize=(20, 20))
    """
    fig.add_subplot(1, 2, 1)
    plt.imshow(img, "gray")
    plt.title('original')
    plt.axis('image') """

    fig.add_subplot(1, 2, 2)
    plt.imshow(invDctImg, "gray")
    plt.title('IDCT')
    plt.axis('image')
    plt.show()

    fig = plt.figure(figsize=(5, 5))
    diffImg = img-invs
    diffImg[diffImg < 0.000001] = 0.

    
    plt.imshow(diffImg, "gray")
    plt.title('original - invDCT')
    plt.axis('image')
    plt.show()

# DCT functions
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
    
def exer(y, cb, cr, block, flag):

    y_dct = blockDct(y, size=block)
    cb_dct= blockDct(cb, size=block)
    cr_dct = blockDct(cr, size=block)
    
    y_dct = np.log(np.abs(y_dct) + 0.0001)
    cb_dct = np.log(np.abs(cb_dct) + 0.0001)
    cr_dct = np.log(np.abs(cr_dct) + 0.0001)

    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.1)
    plt.subplot(1, 3, 1)
    plt.title("DCT " + str(block) + "x" + str(block) + " (Y)")
    plt.imshow(y_dct, "gray")
    plt.subplot(1, 3, 2)
    plt.title("DCT " + str(block) + "x" + str(block) + " (Cb)")
    plt.imshow(cb_dct, "gray")
    plt.subplot(1, 3, 3)
    plt.title("DCT " + str(block) + "x" + str(block) + " (Cr)")
    plt.imshow(cr_dct, "gray")
    plt.show()

    return y_dct, cb_dct, cr_dct

def inv_exer(y_dct, cb_dct, cr_dct, block, flag):

    y_inv = blockDct(y_dct, size=block)
    cb_inv = blockDct(cb_dct, size=block)
    cr_inv = blockDct(cr_dct, size=block)
    
    y_inv = np.log(np.abs(y_inv) + 0.0001)
    cb_inv = np.log(np.abs(cb_inv) + 0.0001)
    cr_inv = np.log(np.abs(cr_inv) + 0.0001)
    
    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.1)
    plt.subplot(1, 3, 1)
    plt.title("Inverted DCT " + str(block) + "x" + str(block) + " (Y)")
    plt.imshow(y_inv, "gray")
    plt.subplot(1, 3, 2)
    plt.title("Inverted DCT " + str(block) + "x" + str(block) + " (Cb)")
    plt.imshow(cb_inv, "gray")
    plt.subplot(1, 3, 3)
    plt.title("Inverted DCT " + str(block) + "x" + str(block) + " (Cr)")
    plt.imshow(cr_inv, "gray")
    plt.show()

    return y_inv, cb_inv, cr_inv

   
    
#𝐈𝐭'𝐬 𝐚 𝐰𝐢𝐥𝐝 𝐩𝐨𝐩𝐩𝐥𝐢𝐞𝐫

def encoder(image):
    img = plt.imread(image +".bmp")
    shape = img.shape
    
    r, g, b = separateRGB(image, flag = True)
    r, g, b = padding(r, g, b, flag = True)
    y , cb, cr, = RGBtoYCrCb(image, flag = True)
    cbStep, crStep, SubCb, SubCr = subsampling(cb, cr, (4,2,0), True, flag = True)

    #devemos usar só o subcb e subcr na dct2, nao sei oque fazer em relação aos outros dois parametros
    y_dct, cb_dct, cr_dct = dct2(y, subcb, subcr)
    y_dct, cb_dct, cr_dct = exer(y_dct, cb_dct, cr_dct, 8, flag = True)
    
    #returnamos os ycbcr do dct e a shape da imagem original
    return  y_dct, cb_dct, cr_dct, shape       
    
        
def decoder(y_dct, cb_dct, cr_dct, shape):
    y, cb, cr = inv_exer(y_dct, cb_dct, cr_dct, 8, flag = True)
    #y, cb, cr = upsampler(y, cb, cr, True, True, True)
    r, g, b = YCbCrtoRGB(y, cb, cr, flag = True)
    rgb = joinRGB(r, g, b)
    rgb = unpad(rgb, shape)
    
    return


def main():

    plt.close('all')

    image = "imagens/barn_mountains"
    y_dct, cb_dct, cr_dct, shape = encoder(image)
    decoder(y_dct, cb_dct, cr_dct, shape)  



if __name__ == "__main__":
    main()



'''
colormap references: https://matplotlib.org/1.2.1/mpl_examples/pylab_examples/show_colormaps.pdf
'''