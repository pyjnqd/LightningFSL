from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
class Chest(Dataset):
    def __init__(self, csv_path, \
        image_path, mode = "test"):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.img_path = image_path
        self.csv_path = csv_path
        self.used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]

        self.labels_maps = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4, "Nodule": 5,  "Pneumothorax": 6}
        
        labels_set = []

        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.labels_all = np.asarray(self.data_info.iloc[:, 1])

        self.image_name  = []
        self.label = []
        # self.transform = transforms.Compose([
        #         transforms.Resize([480,480]),
        #         # transforms.CenterCrop(image_size),
        #         # transforms.Resize([84,84]),

        #         transforms.ToTensor(),
        #         transforms.Normalize(np.array([0.5, 0.5, 0.5]),
        #                                 np.array([0.5, 0.5, 0.5]))])
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        # self.transform = transforms.Compose([
        #         # transforms.Resize([480,480]),
        #         transforms.Resize([256, 256]),
        #         transforms.CenterCrop(224),

        #         transforms.ToTensor(),
        #         normalize])
        # self.transform = transforms.Compose([
        #         transforms.Resize([256, 256]),
        #         transforms.CenterCrop(225),
        #         # transforms.Resize([84,84]),

        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])])
        self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(84),
                # transforms.Resize([84,84]),

                transforms.ToTensor(),
                transforms.Normalize(np.array([0.4712, 0.4499, 0.4031]),
                                        np.array([0.2726, 0.2634, 0.2794]))])

        for name, label in zip(self.image_name_all,self.labels_all):
            label = label.split("|")

            if len(label) == 1 and label[0] != "No Finding" and label[0] != "Pneumonia" and label[0] in self.used_labels:
                self.label.append(self.labels_maps[label[0]])
                self.image_name.append(name)
    
        self.data_len = len(self.image_name)

        self.image_name = np.asarray(self.image_name)
        self.label = np.asarray(self.label)        

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_name[index]

        # Open image
        image = Image.open(self.img_path +  single_image_name).convert('RGB')
        image = self.transform(image)
        # img_as_img = Image.open(self.img_path +  single_image_name).resize((256, 256)).convert('RGB')
        # img_as_img.load()

        # Transform image to tensor
        #img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label[index]

        return (image, single_image_label)

    def __len__(self):
        return self.data_len

def return_class():
    return Chest