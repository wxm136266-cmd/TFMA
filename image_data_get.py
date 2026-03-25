import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
_dir = "E:/graduate/data/cwru/CRWU/"

image_Normal = "./data_SAC/normal5/"
image_Afault = "./data_SAC/AFault5/"
image_Pfault = "./data_SAC/PFault-3channel/"
image_APfault = "./data_SAC/APFault-3channel/"
# new_path = './data_SAC/NBB15_out/'
new_path = './data_SAC/NBB15-N2ST-0101-out/'
sup_path = './data_SAC/normal_color_sup/'

case_SAC_4way = [image_Normal,image_Afault,image_Pfault,image_APfault]

img_size = (64,64)
img_h = 84
img_w = 84
nc = 4
example = 350
chn = 3

def data_generate1():
    x = []
    y = []
    z = []
    for way in case_SAC_4way:
        for filename in os.listdir(way):
            img_path = os.path.join(way, filename)
            # img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(img_path)
            # print(img.shape)

            img = cv2.resize(img, (img_h, img_w))  # [H,W]
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # img = img / 255.
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            img = img.reshape(img_h,img_w,chn)
            x.append(img)
        print(np.shape(x))
        y.append(x)
        print(np.array(y).shape)
        x=[]
    # y = np.array(y).reshape(nc,example,img_h,img_w,chn)
    y=np.array(y)


    np.save('./data_SAC/SAC_img_cwt_84_3channels',y)

def data_generate2():
    x = []
    y = []
    z = []
    i = 0
    for way in case_SAC_4way:

        for filename in os.listdir(way):
            img_path = os.path.join(way, filename)
            img = cv2.imread(img_path)
            # print(img.shape)

            img = cv2.resize(img, (img_h, img_w))  # [H,W]
            img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # img = img / 255.
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            img = img.reshape(img_h,img_w,chn)
            x.append(img)
        print(np.shape(x))
        save_path = f'class{i}_84_3channels_30'
        np.save('./data_SAC/'+save_path,x)

        x=[]
        i = i+1

def data_generate_new():
    x = []
    i = 0
    filenames = [entry.name for entry in os.scandir(new_path) if entry.is_file()]
    filenames = sorted(filenames, key=lambda x: int(x.split('.')[0]))
    print(filenames)


    for filename in filenames:
        img_path = os.path.join(new_path, filename)
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_path)
        # print(img.shape)

        img = cv2.resize(img, (img_h, img_w))  # [H,W]
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # img = img / 255.
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        img = img.reshape(img_h, img_w, chn)
        x.append(img)
    save_path = f'new_data_84_3channels_30'
    np.save('./data_SAC/'+save_path,x)
def data_generate_sup():
    x = []
    i = 0
    filenames = [entry.name for entry in os.scandir(sup_path) if entry.is_file()]
    filenames = sorted(filenames, key=lambda x: int(x.split('.')[0]))
    print(filenames)

    for filename in filenames:
        img_path = os.path.join(sup_path, filename)
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_path)
        # print(img.shape)
        # plt.imshow(img, cmap='gray')

        img = cv2.resize(img, (img_h, img_w))  # [H,W]
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # img = img / 255.
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        img = img.reshape(img_h, img_w, chn)
        # plt.show()
        x.append(img)
    save_path = f'new_data_84_3channels_sup'
    np.save('./data_SAC/'+save_path,x)




def SAC_4way_image(way=2,example=50,spilt=30,shuffle=True):

    if not os.path.isfile(('./data_SAC/SAC_img_cwt_84_3channels.npy')):
        data_generate1()

    data_image = np.load('./data_SAC/SAC_img_cwt_84_3channels.npy')
    if shuffle:
        data_image = sample_shuffle(data_image)

    train_data, test_data = data_image[:, :spilt], data_image[:, spilt:]

    # train_data = [train_class0_data,train_class1_data,train_class2_data,train_class3_data]
    # test_data = [test_class0_data,test_class1_data,test_class2_data,test_class3_data]
    return train_data, test_data


def create_set(data):
    ways , num , _, _,_ = data.shape
    ways = 4
    set_x = []
    set_y = []
    for i in range(ways):
        for j in range(num):
            print(j)
            support_x = data[i][j]
            support_y = i
            set_x.append(support_x)
            set_y.append(support_y)

    return set_x,set_y

def create_set_test(data):
    # ways , num , _, _,_ = data.shape
    ways = 4
    num = [100,100,100,100]
    set_x = []
    set_y = []
    for i in range(ways):
        for j in range(num[i]):
            support_x = data[i][j]
            support_y = i
            set_x.append(support_x)
            set_y.append(support_y)

    return set_x,set_y

def sample_shuffle(data):
    np.random.seed(0)
    for k in range(data.shape[0]):
        np.random.shuffle(data[k])
    # np.random.shuffle(data)
    return data


if __name__ == "__main__":
    data_generate_new()
    data_generate_sup()
