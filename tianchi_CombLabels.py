import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os


# Specify the working directory
os.chdir("C://UCD/__Practicum/ImageTagging/tianchi_data/")

# Get labels
labels1 = pd.read_csv("round1_train/Annotations/label.csv", names=["path", "folder", "code"])
labels2 = pd.read_csv("round2_train/Annotations/label.csv", names=["path", "folder", "code"])
labels = pd.concat([labels1, labels2], ignore_index=True)
labels.head()

# Subset annotations
coat_data = labels[labels['folder'] == 'coat_length_labels'].reset_index(drop=True)
collar_data = labels[labels['folder'] == 'collar_design_labels'].reset_index(drop=True)
lapel_data = labels[labels['folder'] == 'lapel_design_labels'].reset_index(drop=True)
neck_data = labels[labels['folder'] == 'neck_design_labels'].reset_index(drop=True)
neckline_data = labels[labels['folder'] == 'neckline_design_labels'].reset_index(drop=True)
pant_data = labels[labels['folder'] == 'pant_length_labels'].reset_index(drop=True)
skirt_data = labels[labels['folder'] == 'skirt_length_labels'].reset_index(drop=True)
sleeve_data = labels[labels['folder'] == 'sleeve_length_labels'].reset_index(drop=True)

# Create lists for each hierarchy - lapel design, neck design
coat = ['Invisible','High Waist Length','Regular Length','Long Length','Micro Length','Knee Length','Midi Length','Anckle&Floor Length']
collar = ['Invisible', 'Shirt Collar', 'Peter Pan', 'Puritan Collar', 'Rib Collar']
lapel = ["Invisible", "Notched", "Collarless", "Shawl Collar", "Plus Size Shawl"]
neck = ["Invisible", "Turtle Neck", "Ruffle Semi-High Collar", "Low Turtle Neck", "Draped Collar"]
neckline = ['Invisible','Strapless Neck','Deep V Neckline','Straight Neck','V Neckline','Square Neckline','Off Shoulder','Round Neckline','Swear Heart Neck','One Shoulder Neckline']
pant = ['Invisible', 'Short Pant', 'Mid Length', '3/4 Length', 'Cropped Pant', 'Full Length']
skirt = ['Invisible', 'Short Length', 'Knee Length', 'Midi Length', 'Ankle Length', 'Floor Length']
sleeve = ['Invisible','Sleeveless','Cup Sleeves','Short Sleeves','Elbow Sleeves','3_4 Sleeves','Wrist Length','Long Sleeves','Extra Long Sleeves']

categories = ['coat','collar','lapel','neck','neckline','pant','skirt','sleeve']

datasets = []
for i in categories:
    datasets.append(f'{i}_data')

## Decode label 
def decode(data, list):
    for i in range(len(data)):
        pos = data.loc[i,'code'].find('y')
        data.loc[i,'label'] = list[pos]

# Decode labels for each category
for i,j in zip(datasets,categories):
    decode(globals()[i],globals()[j])


## Check the distribution of labels
def Distr(data):
    plt.hist(data['label'])
    plt.show()

Distr(lapel_data)
Distr(neck_data)
Distr(sleeve_data)


## Check the graphs of certain label
def Graph(data, label):
    path = os.getcwd()
    tt = data[data['label'] == label]
    for i in range(9):
        plt.subplot(3,3,i+1)
        img = mpimg.imread(path+'/'+tt.iloc[i,0])
        imgplot = plt.imshow(img)
    plt.show()

Graph(lapel_data, 'Invisible')
Graph(lapel_data, "Collarless")
Graph(neck_data, "Turtle Neck")
Graph(neck_data, "Low Turtle Neck")
Graph(neck_data, "Draped Collar")
Graph(neck_data, "Ruffle Semi-High Collar")
Graph(sleeve_data, "Invisible")
Graph(sleeve_data, "Short Sleeves")
Graph(sleeve_data, "Wrist Length")

## Combine labels
'''lapel: Collarless -> Invisible; Plus Size Shawl -> Shawl Collar'''
lapel_data = lapel_data.replace(["Collarless","Plus Size Shawl"], ['Invisible',"Shawl Collar"])
lapel_data.head()
lapel_data['label'].unique()

'''neck: Low Turtle Neck -> Turtle Neck'''
neck_data = neck_data.replace(["Low Turtle Neck"], ["Turtle Neck"])
neck_data.head()
neck_data['label'].unique()

'''sleeve: Drop Invisible; Extra Long Sleeves, Wrist Length -> Long Sleeves; 3/4 Sleeves -> Elbow Sleeves; Cup Sleeves -> Short Sleeves'''
sleeve_data = sleeve_data[sleeve_data['label'] != 'Invisible']
sleeve_data = sleeve_data.replace(["Extra Long Sleeves", "Wrist Length","3_4 Sleeves","Cup Sleeves"], ["Long Sleeves","Long Sleeves","Elbow Sleeves","Short Sleeves"])
sleeve_data.head()
sleeve_data['label'].unique()
sleeve_data.shape