import matplotlib.pyplot as plt
import matplotlib.colors as clr


def main():
    plt.close('all')
    img = plt.imread('imagens\peppers.bmp')
    #plt.figure()
    #plt.imshow(img)
    #plt.show()
    #print(img.shape)
    #print(img.dtype)


    
    R = img[:,:,0]
    #print(R.shape)
    #print(R[0,0])

    cmRed = clr.LinearSegmentedColormap.from_list('myRed', ([0,0,0], [1,0,0]), 86)
    cmGreen = clr.LinearSegmentedColormap.from_list('myRed', ([0,0,0], [0,1,0]), 256)
    cmBlue = clr.LinearSegmentedColormap.from_list('myRed', ([0,0,0], [0,0,1]), 256)
    cmGray = clr.LinearSegmentedColormap.from_list('myGray', ([0,0,0], [1,1,1]), 256)
    
    plt.figure()
    #plt.title('Bunfeecaaa')
    plt.imshow(R, cmRed)
   # plt.figure()
   # plt.imshow(R, cmGray)
    plt.show()




if __name__ == "__main__":
    main()
