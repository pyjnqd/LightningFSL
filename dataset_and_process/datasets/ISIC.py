from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


class ISIC(Dataset):
    def __init__(self, image_path, csv_path, mode):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.img_path = image_path
        self.csv_path = csv_path
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
        if mode == "train":
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.4712, 0.4499, 0.4031]),
                                        np.array([0.2726, 0.2634, 0.2794]))])
        else:
            # self.transform = transforms.Compose([
            #     transforms.Resize([256, 256]),
            #     transforms.CenterCrop(224),
            #     # transforms.Resize([84,84]),

            #     transforms.ToTensor(),
            #     normalize])
            self.transform = transforms.Compose([
                    transforms.Resize([92, 92]),
                    transforms.CenterCrop(84),
                    # transforms.Resize([84,84]),

                    transforms.ToTensor(),
                    transforms.Normalize(np.array([0.4712, 0.4499, 0.4031]),
                                            np.array([0.2726, 0.2634, 0.2794]))])
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name = np.asarray(self.data_info.iloc[:, 0])

        self.label = np.asarray(self.data_info.iloc[:, 1:])

        # print(self.labels[:10])
        self.label = (self.label!=0).argmax(axis=1)
        # Calculate len
        self.data_len = len(self.label)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_name[index]
        # Open image
        image = Image.open(self.img_path +  single_image_name + ".jpg").convert('RGB')
        image = self.transform(image)

        # Transform image to tensor
        #img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label[index]

        return (image, single_image_label)

    def __len__(self):
        return self.data_len


def return_class():
    return ISIC

if __name__ == "__main__":
    a = ISIC("../../../data/cross_domain/ISIC2018_Task3_Training_Input/", "../../../data/cross_domain/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv", "all")
    print(len(a.labels))