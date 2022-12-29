import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# from sklearn.cluster import KMeans
import cv2
#from skimage.color import rgb2lab, deltaE_cie76
from collections import Counter
import os

def RGB_HEX(color):
    hex_val=f"#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}"
    return hex_val

def get_colors_in_image(image, number_of_colors, show_chart):
    image = cv2.imread(image)
    hsvImg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    cv2.imshow('HSV image RGB', hsv)
    reshaped_image = cv2.resize(image, (600, 400))
    reshaped_image = reshaped_image.reshape(reshaped_image.shape[0]*reshaped_image.shape[1], 3)
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(reshaped_image)
    counts = Counter(labels)
    counts = dict(sorted(counts.items()))
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB_HEX(ordered_colors[i]) for i in counts.keys()]
    
#  rgb_colors = [ordered_colors[i] for i in counts.keys()]
    # if (show_chart):
    #     plt.figure(figsize = (8, 6))
    #     plt.pie(counts.values(), labels = hex_colors, colors = hex_colors,autopct='%1.1f%%')
    #     plt.show()
    return hex_colors

def saveImageQuery():
    print("To save, enter image name eg:result.png, result.jpg\nTo cancel enter \"c\"\n")
    value=input()
    if value.lower() == "c":
        plt.show() 
        plt.close()
    else:
        p = "images/"+value
        plt.savefig(p)
        plt.show() 
        plt.close()
        
def colorHistogram(image):
    image = cv2.imread(image)
    color = ['blue', 'green', 'red']
    for i, color in enumerate(color):
        print(i)
        hist = cv2.calcHist([image], [i], None, [256], [0,256])
        plt.subplot(3,1,(i+1))
        plt.plot(hist,color=color)
        plt.xlim([0,256])
    saveImageQuery()

def colorHistogramTogether(image):
    image = cv2.imread(image)
    color = ['blue', 'green', 'red']
    for i, color in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0,256])
        plt.plot(hist,color=color)
        plt.xlim([0,256])
       
    saveImageQuery()
        
def imageBlur(image,show=False):
    ''' Get edges within an image '''
    grey_img = cv2.imread(image)
    blur_img = cv2.GaussianBlur(grey_img,(5,5), 1)
    # canny_img = cv2.Canny(blur_img,100,100)
    if show:
        cv2.imshow("Original image",grey_img)
        cv2.imshow("Blur image",blur_img)
        cv2.waitKey(0)
    return blur_img

def imageSegmentationFilter(image):
    # read image
    image = cv2.imread(image)
    rgbImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    greyImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2GRAY)
    
    # applying otsu_image thresholding
    from skimage.filters import threshold_otsu  
    thresh_val = threshold_otsu(greyImg)

    img_seg = greyImg<thresh_val
    # plt.imshow(img_seg)
    f_image=filter_image(image, img_seg)
    cv2.imshow("Segmented image",f_image)
    cv2.waitKey(0)
    return

def imageSegmentationColor(image):
    lower = np.array([0, 0, 0])
    upper = np.array([252, 255, 130])
    # read image
    image = cv2.imread(image)
    rbgImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    mask = cv2.inRange(rbgImg,lower,upper)
    result = cv2.bitwise_and(rbgImg, rbgImg, mask = mask)
    plt.imshow(result)
    saveImageQuery()
    
    
def filter_image(image, mask):
    
    r = image[:,:,0] * mask
    g = image[:,:,1] * mask
    b = image[:,:,2] * mask

    return np.dstack([r,g,b])

def imageSegmentationEdges(image):
    # read image
    image = cv2.imread(image)
    rgbImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgbImg= cv2.resize(rgbImg,(256,256))
    greyImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2GRAY)
    _,thresh = cv2.threshold(greyImg, np.mean(greyImg), 255, cv2.THRESH_BINARY_INV)
    
    # get edges
    edges = cv2.dilate(cv2.Canny(thresh,0,255),None)
    
    # get contours and mask
    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    mask = np.zeros((256,256), np.uint8)
    masked = cv2.drawContours(mask, [cnt],-1, 255, -1)
    
    # segmentation
    dst = cv2.bitwise_and(rgbImg, rgbImg, mask=masked)
    segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    cv2.imshow("Segmented image",segmented)
    cv2.waitKey(0)
    return


def image_resize():
    # image = cv2.imread(image)
    # rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # img = mpimg.imread(image)
    f = r'E:/BlightProject_CNN/testdata1/data'
    
    directories = os.listdir(f)
    # print(directories)
    resized_directory = 'resized/'
    for directory in directories:
        directory_path = f"{f}/{directory}"
        print(f"At: {directory_path}")
        for file in os.listdir(directory_path):
            # print(file)
            file_path = f"{directory_path}/{file}"
            # print(file_path)
            img = cv2.imread(file_path)
            # print(img)
            scale_percent = 15 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            # Create resized image using the calculated dimensions
            resized_image = cv2.resize(img, dim,interpolation=cv2.INTER_AREA)
        
            # Save the image in Output Folder
            save_to = resized_directory+directory+'/resized_'+str(width)+'_'+str(height)+file
            print(save_to)
            cv2.imwrite(resized_directory+directory+'/resized_'+str(width)+'_'+str(height)+file,resized_image)
    return dim

if __name__=="__main__":
    
    instructions = "Select options below\n 1. Image Resizing\n 2. Color histogram together\n 3. Color histogram separate\n 4. Image segmentation with color\n\n"
    choice = int(input(instructions))
    
    if choice == 1:
        final_dim=image_resize()
        print(final_dim)
    elif choice == 2:
        colorHistogramTogether("images/DSC_0003.JPG")
    elif choice == 3:
        colorHistogram("images/DSC_0003.JPG")
    elif choice == 4:
        imageSegmentationColor("images/DSC_0006.JPG")
    else:
        print("Your choice not valid")
        import sys
        sys.exit()