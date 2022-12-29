import urllib.request
import zipfile
import os
import cv2


class ProcessingUtils():
          
    def zipfileDownload(self,url, ref_directory,file_name):
        '''
        Download zipfile containining image data
        url: link to zipfile
        file_name: name to save the file to
        '''
        print("Starting file download ...\n")
        file_name = os.path.join(ref_directory,file_name)
        res=urllib.request.urlretrieve(url,file_name)
        if res:
            print("Completed downloading ", res[0])
            return file_name
        print(f"Unsuccessful download!\n From link {url} to file {file_name}")
        return False

    def readZipFile(self,file_name,target_folder,ref_directory):
        '''
        read zipfile contents into a folder
        '''
        file_name = os.path.join(ref_directory,file_name)
        print(f"Starting {file_name} file read")
        zip_ref = zipfile.ZipFile(file_name,'r')
        target_folder = os.path.join(ref_directory,target_folder)
        zip_ref.extractall(path=target_folder)
        zip_ref.close()
        print(f"Completed {file_name} file read into {target_folder}")
        return target_folder

    def saveModel(self,ref_directory,model):
          '''
          Save tensorflow model to directory "saved_models"
          model: receives tensorflow model as input
          '''
          save_model_folder = os.path.join(ref_directory,"saved_models")
          next_version=len(os.listdir(save_model_folder))+1
          name = f"saved_models/tomatoe_blight_model_version_{next_version}.h5"
          print(os.path.join(ref_directory,name))
          model.save(name)
          return

    def getLoadModel(self,ref_directory):
        '''
        Loads saved tensorflow model from directory "saved_models"
        model: receives tensorflow model as input
        returns a tensorflow model
        '''
        from tensorflow.keras.models import load_model
        model_path = os.path.join(ref_directory,"saved_models")

        latest_version =len(os.listdir(model_path))
        name_hfile = f"tomatoe_blight_model_version_{latest_version}.h5"
        name_hfile_model_path = os.path.join(model_path,name_hfile)
        print(name_hfile_model_path)
        loaded_model=load_model(name_hfile_model_path)
        return loaded_model
    
    
    def image_resize(self,ref_directory,raw_data_path):
                     
        f = os.path.join(ref_directory,raw_data_path)
        
        directories = os.listdir(f)
        # print(directories)
        resized_directory = os.path.join(ref_directory,'resized/')
        for directory in directories:
            directory_path = f"{f}/{directory}"
            print(f"At: {directory_path}")
            for file in os.listdir(directory_path):
                # print(file)
                file_path = f"{directory_path}/{file}"
                # print(file_path)
                img = cv2.imread(file_path)
                # print(img)
                scale_percent = 10 # percent of original size
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim = (width, height)
                
                # Create resized image using the calculated dimensions
                resized_image = cv2.resize(img, dim,interpolation=cv2.INTER_AREA)
            
                # Save the image in Output Folder
                save_to = resized_directory+directory+'/resized_'+str(width)+'_'+str(height)+file
                print(save_to)
                cv2.imwrite(resized_directory+directory+'/resized_'+str(width)+'_'+str(height)+file,resized_image)
    # color analysis
    # https://pyimagesearch.com/2021/04/28/opencv-color-spaces-cv2-cvtcolor/
    # https://www.projectpro.io/recipes/detect-specific-colors-from-image-opencv
        
        # image segmentation 
    # https://www.kaggle.com/code/sanikamal/image-segmentation-using-color-spaces
    # https://medium.com/srm-mic/color-segmentation-using-opencv-93efa7ac93e2
    # https://mattmaulion.medium.com/color-image-segmentation-image-processing-4a04eca25c0