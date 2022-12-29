

def image_resize(raw_data_path):
    # image = cv2.imread(image)
    # rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # img = mpimg.imread(image)
    import os
    import cv2
    
    f = raw_data_path
    
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