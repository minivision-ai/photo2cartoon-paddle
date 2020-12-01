import os
import cv2
from paddle.io import Dataset


class ImageFolder(Dataset):
    def __init__(self, data_path, image_size=256, transform=None):
        super(ImageFolder, self).__init__()

        self.img_names = [data_path+'/'+x for x in os.listdir(data_path)]
        self.image_size = image_size
        self.transform = transform

    def __getitem__(self, idx):
        train_image = cv2.imread(self.img_names[idx])
        train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)
        train_image = self.transform(train_image)
        return train_image, 0

    def __len__(self):
        return len(self.img_names)


if __name__ == '__main__':
    from paddle.vision.transforms import transforms as T
    from paddle.io import DataLoader

    img_size = 256
    pad = 30

    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.Resize((img_size + pad, img_size + pad)),
        T.RandomCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=0.5, std=0.5)
    ])

    dataloader = ImageFolder('dataset/photo2cartoon/trainB', transform=train_transform)

    train_loader = DataLoader(dataloader, batch_size=1, shuffle=True)
    print('num: ', len(train_loader))
    for i in range(300):
        print(i)
        try:
            real_A, _ = next(trainA_iter)
        except:
            trainA_iter = iter(train_loader())
            real_A, _ = next(trainA_iter)
        print(real_A.shape)
    '''
    for data in dataloader:
        print(data)
        train_image = data * 127.5 + 127.5
        train_image = train_image.numpy().transpose(1,2,0).astype(np.uint8)[:,:,::-1]
        cv2.imwrite('test.png', train_image)
    '''
