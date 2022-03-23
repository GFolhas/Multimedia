from PIL import Image
from scipy import fftpack as fft
from scipy import ifft
from matplotlib import cm

import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import cv2
import math


Tc = np.array([[0.299, 0.587, 0.114],[-0.168736, -0.331264, 0.5],[0.5, -0.418688, -0.081312]])
TcInverted = np.linalg.inv(Tc)


QY = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68,  109, 103, 77],
    [24, 35, 55, 64, 81,  104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]])


QC = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]])



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

    image = plt.imread(images + ".bmp")

    #print(image)
    red = image[:,:,0]
    green = image[:,:,1]
    blue = image[:,:,2]

    #print(image.shape) -> (384, 512, 3)

    RED = clr.LinearSegmentedColormap.from_list('cmap', [(0, 0, 0), (1,0,0)], N=256)
    GREEN = clr.LinearSegmentedColormap.from_list('cmap', [(0, 0, 0), (0,1,0)], N=256)
    BLUE = clr.LinearSegmentedColormap.from_list('cmap', [(0, 0, 0), (0,0,1)], N=256)
    
    if flag:
        plt.subplots_adjust(left=0.01, right=0.99, wspace=0.1)

        plt.subplot(1, 3, 1)
        plt.title("Red Channel")
        plt.imshow(red, RED)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Green Channel")
        plt.imshow(green, GREEN)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("Blue Channel")
        plt.imshow(blue, BLUE)    
        plt.axis('off')
        
        plt.show()

    return red, green, blue
    
# Join RGB
def joinRGB(r, g, b, flag):
    
    shape = (r.shape[0], r.shape[1], 3)
    rgb = np.empty(shape, dtype=r.dtype)
    rgb[:,:,0] = r
    rgb[:,:,1] = g
    rgb[:,:,2] = b    
 
    #rgb = rgb.astype(np.uint8)
    
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
    c_shape = 16 - countC if countC != 0 else 0
    h_shape = 16 - countH if countH != 0 else 0
    if (h_shape) > 0:
        r = np.vstack((r, np.tile(r[-1, :], (h_shape, 1))))
        g = np.vstack((g, np.tile(g[-1, :], (h_shape, 1))))
        b = np.vstack((b, np.tile(b[-1, :], (h_shape, 1))))
        h, c = r.shape
    
    if (c_shape) > 0:
        r = np.hstack((r, np.tile(r[:, -1], (c_shape, 1)).T))
        g = np.hstack((g, np.tile(g[:, -1], (c_shape, 1)).T))
        b = np.hstack((b, np.tile(b[:, -1], (c_shape, 1)).T))
        h, c = r.shape
    
    paddedImg = np.zeros((h, c, 3))
    paddedImg[:, :, 0] = r
    paddedImg[:, :, 1] = g
    paddedImg[:, :, 2] = b
    print("dim = ", paddedImg.shape)
    paddedImg = paddedImg.astype(np.uint8)
    
    if flag:
        plt.figure()
        plt.title("padded")
        plt.imshow(paddedImg)
        plt.axis('off')
        plt.show()
    
    return r, g, b

# Unpadding    
def unpad(paddedImg, shape, flag):
    unpaddedImg = paddedImg[:shape[0], :shape[1], :]
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
def RGBtoYCrCb(r, g, b, flag):
    # Flag para fzr plots

    #No more np.dot
    #No more ycbcr matrix
    #No more fortnight

    y = Tc[0, 0] * r + Tc[0, 1] * g + Tc[0, 2] * b
    cb = Tc[1, 0] * r + Tc[1, 1] * g + Tc[1, 2] * b + 128
    cr = Tc[2, 0] * r + Tc[2, 1] * g + Tc[2, 2] * b + 128

    if flag:
        plt.subplots_adjust(left=0.01, right=0.99, wspace=0.1)

        plt.subplot(1, 3, 1)
        plt.title("YCbCr from RBG Image (channel - Y)")
        plt.imshow(y, "gray")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("YCbCr from RBG Image (channel - Cb)")
        plt.imshow(cb, "gray")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("YCbCr from RBG Image (channel - Cr)")
        plt.imshow(cr, "gray")    
        plt.axis('off')
        
        plt.show()

    return y, cb, cr

# YCbCr to RGB
def YCbCrtoRGB(y: np.ndarray, cb: np.ndarray, cr: np.ndarray, shape, flag): 
    cb -= 128
    cr -= 128

    r = TcInverted[0, 0] * y + TcInverted[0, 1] * cb + TcInverted[0, 2] * cr 
    g = TcInverted[1, 0] * y + TcInverted[1, 1] * cb + TcInverted[1, 2] * cr
    b = TcInverted[2, 0] * y + TcInverted[2, 1] * cb + TcInverted[2, 2] * cr

    rgb = joinRGB(r, g, b, flag = False) 
    rgb = np.round(rgb)
    rgb[rgb > 255] = 255
    rgb[rgb < 0] = 0

    rgb = rgb.astype(np.uint8)
    
    if flag:
        plt.figure()
        plt.title("RGB from YCbCr Image")
        plt.imshow(rgb)
        plt.show()
    
    return rgb

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

        return  dsCbInterp, dsCrInterp

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

        return  cbDown, crDown

# UpSampling
def upsampler(dsCb, dsCr, shape, inter, flag):

    # Flag inter
    size = shape[::-1]

    if inter:
        interpolation = cv2.INTER_LINEAR
        name = "interpolation"
    else:
        interpolation = cv2.INTER_AREA
        name = "repetition"
    usCb = cv2.resize(dsCb, size, interpolation)
    usCr = cv2.resize(dsCr, size, interpolation)

    if flag:
        plt.subplots_adjust(left=0.01, right=0.99, wspace=0.1)
        plt.subplot(1, 2, 1)
        plt.title("Cb Upsampled with " + name)
        plt.imshow(usCb, "gray")
        plt.subplot(1, 2, 2)
        plt.title("Cr Upsampled with " + name)
        plt.imshow(usCr, "gray")
        plt.show()

    return usCb, usCr

# Exercise 7 - DCT and Block DCT

# DCT
def fullDct(y, cb, cr, flag):

    y_dct = dct(y)
    cb_dct = dct(cb)
    cr_dct = dct(cr)

    y_dct = np.log(np.abs(y_dct) + 0.0001)
    cb_dct = np.log(np.abs(cb_dct) + 0.0001)
    cr_dct = np.log(np.abs(cb_dct) + 0.0001)

    if flag:
        plt.subplot(1, 3, 1)
        plt.imshow(y_dct, "gray")
        plt.title('DCT - y')
        plt.axis('image')                       

        plt.subplot(1, 3, 2)
        plt.imshow(cb_dct, "gray")
        plt.title('DCT - cb')
        plt.axis('image')

        plt.subplot(1, 3, 3)
        plt.imshow(cr_dct, "gray")
        plt.title('DCT - cr')
        plt.axis('image')

        plt.show() 

    return y_dct, cb_dct, cr_dct

# IDCT

def fullIdct(ydct, cbdct, crdct, shape):

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

# Block DCT
def dct_block(y, cb, cr, block, flag):

    y_dct = blockDct(y, size=block)
    cb_dct= blockDct(cb, size=block)
    cr_dct = blockDct(cr, size=block)
    
    y_block = np.log(np.abs(y_dct) + 0.0001)
    cb_block = np.log(np.abs(cb_dct) + 0.0001)
    cr_block = np.log(np.abs(cr_dct) + 0.0001)

    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.1)
    plt.subplot(1, 3, 1)
    plt.title("DCT " + str(block) + "x" + str(block) + " (Y)")
    plt.imshow(y_block, "gray")
    plt.subplot(1, 3, 2)
    plt.title("DCT " + str(block) + "x" + str(block) + " (Cb)")
    plt.imshow(cb_block, "gray")
    plt.subplot(1, 3, 3)
    plt.title("DCT " + str(block) + "x" + str(block) + " (Cr)")
    plt.imshow(cr_block, "gray")
    plt.show()

    return y_dct, cb_dct, cr_dct

def idct_block(y_dct, cb_dct, cr_dct, block, flag):

    y_inv = blockIdct(y_dct, size=block)
    cb_inv = blockIdct(cb_dct, size=block)
    cr_inv = blockIdct(cr_dct, size=block)
    
    #y_inv = np.log(np.abs(y_inv) + 0.0001)
    #cb_inv = np.log(np.abs(cb_inv) + 0.0001)
    #cr_inv = np.log(np.abs(cr_inv) + 0.0001)
    
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


def quantizer(ycbcr: tuple, qf: int):
    y, cb, cr = ycbcr
    sf = (100 - qf) / 50 if qf >= 50 else 50 / qf
    QsY = np.round(QY * sf)
    QsC = np.round(QC * sf)

    QsY[QsY > 255] = 255
    QsC[QsC > 255] = 255
    QsY[QsY < 1] = 1
    QsC[QsC < 1] = 1

    qy = np.empty(y.shape, dtype=y.dtype)
    qcb = np.empty(cb.shape, dtype=cb.dtype)
    qcr = np.empty(cr.shape, dtype=cr.dtype)

    for i in range(0, y.shape[0], 8):
        for j in range(0, y.shape[1], 8):
            qy[i:i+8, j:j+8] = y[i:i+8, j:j+8] / QsY
    qy = np.round(qy)

    for i in range(0, cb.shape[0], 8):
        for j in range(0, cb.shape[1], 8):
            qcb[i:i+8, j:j+8] = cb[i:i+8, j:j+8] / QsC
    qcb = np.round(qcb)

    for i in range(0, cr.shape[0], 8):
        for j in range(0, cr.shape[1], 8):
            qcr[i:i+8, j:j+8] = cr[i:i+8, j:j+8] / QsC
    qcr = np.round(qcr)


    ly = np.log(np.abs(qy) + 0.0001)
    lcb = np.log(np.abs(qcb) + 0.0001)
    lcr = np.log(np.abs(qcr) + 0.0001)
    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.1)
    plt.subplot(1, 3, 1)
    plt.title("Quantized (Y) channel w/ " + str(qf) + " quality factor")
    plt.imshow(ly, "gray")
    plt.subplot(1, 3, 2)
    plt.title("Quantized (Cb) channel w/ " + str(qf) + " quality factor")
    plt.imshow(lcb, "gray")
    plt.subplot(1, 3, 3)
    plt.title("Quantized (Cr) channel w/ " + str(qf) + " quality factor")
    plt.imshow(lcr, "gray")
    plt.show()

    return qy, qcb, qcr

def iQuantizer(ycbcr: tuple, qf: int):
    qy, qcb, qcr = ycbcr
    sf = (100 - qf) / 50 if qf >= 50 else 50 / qf
    QsY = np.round(QY * sf)
    QsC = np.round(QC * sf)

    QsY[QsY > 255] = 255
    QsC[QsC > 255] = 255
    QsY[QsY < 1] = 1
    QsC[QsC < 1] = 1

    y = np.empty(qy.shape, dtype=qy.dtype)
    cb = np.empty(qcb.shape, dtype=qcb.dtype)
    cr = np.empty(qcr.shape, dtype=qcr.dtype)

    for i in range(0, y.shape[0], 8):
        for j in range(0, y.shape[1], 8):
            y[i:i+8, j:j+8] = qy[i:i+8, j:j+8] * QsY

    for i in range(0, cb.shape[0], 8):
        for j in range(0, cb.shape[1], 8):
            cb[i:i+8, j:j+8] = qcb[i:i+8, j:j+8] * QsC

    for i in range(0, cr.shape[0], 8):
        for j in range(0, cr.shape[1], 8):
            cr[i:i+8, j:j+8] = qcr[i:i+8, j:j+8] * QsC

    ly = np.log(np.abs(qy) + 0.0001)
    lcb = np.log(np.abs(qcb) + 0.0001)
    lcr = np.log(np.abs(qcr) + 0.0001)
    plt.subplots_adjust(left=0.01, right=0.99, wspace=0.1)
    plt.subplot(1, 3, 1)
    plt.title("iQuantized (Y) channel w/ " + str(qf) + " quality factor")
    plt.imshow(ly, "gray")
    plt.subplot(1, 3, 2)
    plt.title("iQuantized (Cb) channel w/ " + str(qf) + " quality factor")
    plt.imshow(lcb, "gray")
    plt.subplot(1, 3, 3)
    plt.title("iQuantized (Cr) channel w/ " + str(qf) + " quality factor")
    plt.imshow(lcr, "gray")
    plt.show()

    return y, cb, cr


# Exercise 9 - DPCM

def DPCM(imgDCT_Q):

    #DPCM 8x8
    imgDPCM = imgDCT_Q.copy()
    dc0 = imgDPCM[0,0]
    nl, nc = imgDPCM.shape
    for i in range(0, nl, 8):
        for j in range(0, nc, 8):
            if i == 0 and j == 0:
                #dc0 = imgDCT_Q[i, j]
                continue
            dc = imgDCT_Q[i, j]
            diff = dc - dc0
            dc0 = dc
            imgDPCM[i, j] = diff

    fig = plt.figure(figsize=(15, 15))
    plt.imshow(np.log(np.abs(imgDPCM) + 0.0001), "gray")
    plt.title('DPCM')
    plt.xticks([])
    plt.yticks([])
    plt.axis('image')

    plt.show()



def iDPCM(imgDCT_Q):

    #DPCM 8x8
    imgDPCM = imgDCT_Q.copy()
    dc0 = imgDPCM[0,0]
    nl, nc = imgDPCM.shape
    for i in range(0, nl, 8):
        for j in range(0, nc, 8):
            if i == 0 and j == 0:
                continue
            dc = imgDCT_Q[i, j]
            s = dc + dc0
            dc0 = dc
            imgDPCM[i, j] = s

    fig = plt.figure(figsize=(15, 15))
    plt.imshow(np.log(np.abs(imgDPCM) + 0.0001), "gray")
    plt.title('iDPCM')
    plt.xticks([])
    plt.yticks([])
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

   
    
#𝐈𝐭'𝐬 𝐚 𝐰𝐢𝐥𝐝 𝐩𝐨𝐩𝐩𝐥𝐢𝐞𝐫

def encoder(image, quality):
    img = plt.imread(image +".bmp")
    shape = img.shape
    
    r, g, b = separateRGB(image, flag = True)
    r, g, b = padding(r, g, b, flag = True)
    y , cb, cr, = RGBtoYCrCb(r, g, b, flag = True)
    SubCb, SubCr = subsampling(cb, cr, (4,2,0), inter = True, flag = True)

    #devemos usar só o subcb e subcr na dct2, nao sei oque fazer em relação aos outros dois parametros
    #y_dct, cb_dct, cr_dct = fullDct(y, SubCb, SubCr, flag = True)
    y_block, cb_block, cr_block = dct_block(y, SubCb, SubCr, block = 8, flag = True)

    qy, qcb, qcr = quantizer((y_block, cb_block, cr_block), quality)

    DPCM(qy)
    DPCM(qcb)
    DPCM(qcr)
    
    #returnamos os ycbcr do dct e a shape da imagem original
    return  qy, qcb, qcr, shape, quality    
    

def decoder(qy, qcb, qcr, shape, quality):
    
    iDPCM(qy)
    iDPCM(qcb)
    iDPCM(qcr)

    y_dct, cb_dct, cr_dct = iQuantizer((qy,qcb,qcr), quality)
    y, cb, cr = idct_block(y_dct, cb_dct, cr_dct, block = 8, flag = True)

    cb, cr = upsampler(cb, cr, y.shape, True, flag = True)
    rgb = YCbCrtoRGB(y, cb, cr, shape, flag = True)
    rgb = unpad(rgb, shape, flag = True)
    
    return rgb

def MSE(original, comp, height, width):
    original = original.astype(np.float64)
    comp = comp.astype(np.float64)
    coef = 1 / (height * width)
    sums = np.sum((original - comp) ** 2)
    return coef * sums


def RMSE(mse):
    return math.sqrt(mse)

# THIS IS ONLY GIVING ME HALF OF THE EXPECTED RESULT FOR SOME REASON
def SNR(original, mse, height, width, flag=None):
    if flag:
        original = original.astype(np.float64)
        coef =  1 / (height * width)
        sums = np.sum(original ** 2)
        return 10 * np.log10((coef * sums) / mse)
    else:
        maxsqr = np.max(original)**2
        return 10 * np.log10(maxsqr / mse)


# Exercise 10 - Compression
def compression(compressed):
    original = plt.imread("imagens/barn_mountains.bmp")
    height, width = original[:,:,0].shape
    mse = MSE(original, compressed, height, width)
    rmse = RMSE(mse)
    snr = SNR(original, mse, height, width, 1)
    psnr = SNR(original, mse, height, width)   
    showCompValues(mse, rmse, snr, psnr)

def showCompValues(mse, rmse, snr, psnr):
    print("MSE: " + str(mse))
    print("RMSE: " + str(rmse))
    print("SNR: " + str(snr))
    print("PSNR: " + str(psnr))


def main():

    plt.close('all')

    image = "imagens/barn_mountains"
    quality = 50
    y_dct, cb_dct, cr_dct, shape, quality = encoder(image, quality)
    img = decoder(y_dct, cb_dct, cr_dct, shape, quality)
    return img



if __name__ == "__main__":
    reconstructed = main()
    compression(reconstructed)



'''
colormap references: https://matplotlib.org/1.2.1/mpl_examples/pylab_examples/show_colormaps.pdf
'''