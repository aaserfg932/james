import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader,Dataset#此处加入dataset
from torch.autograd import Variable
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.font_manager import FontProperties
import pandas as pd


model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)
os.makedirs("images_WGAN", exist_ok=True)#输出图片保存路径

parser = argparse.ArgumentParser()#设置超参数
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lrg", type=float, default=0.00004, help="learning rate of generator")
parser.add_argument("--lrd", type=float, default=0.00002, help="learning rate of discriminator")
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")#噪声向量的维度
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")#输入图片的尺寸
parser.add_argument("--channels", type=int, default=3, help="number of image channels")#输入图片的通道数
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)#图片格式

cuda = torch.cuda.is_available()
print(cuda)

current_dir = os.getcwd()

class Generator(nn.Module):#生成器网络模型
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]#设置全连接层
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))#归一化处理，加快训练速度
            layers.append(nn.LeakyReLU(0.2, inplace=True))#加入LeakyReLU激活函数
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img

G = Generator()
# print(G)

class Discriminator(nn.Module):#判别器网络模型
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

D = Discriminator()
# print(D)
#
# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

img_transform=transforms.Compose([
    transforms.Resize((256, 256)),#图片进行裁剪
    transforms.ToTensor(),#图片转换为tensor格式
    transforms.Normalize((0.5,),(0.5,))#图片进行标准化处理
])
class MyData(Dataset):#设置数据集
    def __init__(self,root_dir,transform=None):
        self.root_dir=root_dir#文件目录
        self.transform=transform#图像变换
        self.images=os.listdir(self.root_dir)#遍历目录下的图片文件

    def __len__(self):
        return len(self.images)#返回数据集中所有图片的个数

    def __getitem__(self, index):
        image_index=self.images[index]
        image_path=os.path.join(self.root_dir,image_index)
        image=Image.open(image_path).convert('RGB')
        if self.transform:
            image=self.transform(image)
        return image
mydataset=MyData(
    root_dir='tdata/test_data',transform=img_transform
)

dataloader=DataLoader(
    dataset=mydataset,batch_size=opt.batch_size,shuffle=True
)



# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lrg)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lrd)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

G_losses = []
D_losses = []

batches_done = 0
for epoch in range(opt.n_epochs):

    for i,imgs in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))#判别器损失函数, Wasserstein distance

        loss_D.backward()
        optimizer_D.step()



        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)
            # flattened_tensor = torch.flatten(gen_imgs, start_dim=1)
            # scaled_tensor = (flattened_tensor + 1) / 2
            # one_dimensional_tensor = torch.flatten(scaled_tensor)
            # input_data = one_dimensional_tensor.detach().cpu().numpy()
            # print(input_data)                                         #將生成器產生的張量轉換為numpy數據

            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))#生成器损失函数

            loss_G.backward()
            optimizer_G.step()

            D_losses.append(abs(loss_D.item()))
            G_losses.append(abs(loss_G.item()))

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
            )

        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:1], "images_WGAN/%d.png" % batches_done, nrow=1, normalize=True)
        if batches_done % (opt.sample_interval) * 10 == 0:
            torch.save(generator.state_dict(), os.path.join(model_dir, "generator.pth"))
            torch.save(discriminator.state_dict(), os.path.join(model_dir, "discriminator.pth"))
        batches_done += 1
        if batches_done == opt.n_epochs * len(dataloader):  # 在所有epoch训练完成后保存损失值
            # Convert lists to pandas DataFrame
            losses_df = pd.DataFrame({'D_loss': D_losses, 'G_loss': G_losses})

            # Save DataFrame to Excel file
            losses_df.to_excel('losses1.xlsx', index=False)

# plt.rcParams['font.family'] = 'Times New Roman'
# plt.figure(figsize = (10, 8))
# plt.title("Generator and Discriminator Loss During Training", fontsize=25)
# plt.plot(D_losses, label='D loss')
# plt.plot(G_losses, label='G loss')
# plt.xlabel('Epoch', fontsize=25)
# plt.ylabel('Loss', fontsize=25)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.legend()
# #
# font_props = FontProperties(size=20, family='Times New Roman')
# plt.legend(prop=font_props)
# plt.savefig('loss_plot.svg', format='pdf', dpi=400)
# plt.show()