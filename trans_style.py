import torch
from PIL import Image
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models

# 设置图片大小
img_size = 512

# 设置GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.229, 0.224, 0.225])

# 加载图片
def load_img(img_path):
    img = Image.open(img_path).convert('RGB') # 将RGBA转为RGB
    img = img.resize((img_size, img_size))
    img = transforms.ToTensor()(img)
    img = transform(img).unsqueeze(0) # 升为[batch_size, chanal, hight, width]
    img = img.to(device)
    return img

# 显示图片
def show_img(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    return image

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(pretrained = True).features  # 提取卷积层

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


# 加载图片
content_img = load_img("./input/golden_gate.jpg")
style_img = load_img("./input/starry_night.jpg")

# 生成图片
target_img = content_img.clone().requires_grad_(True)

# 优化器
optimizer = torch.optim.Adam([target_img], lr=0.003)
vgg = VGGNet().to(device).eval()
total_step = 2000 # 训练次数
style_weight = 8000 # style_loss的权重

writer = SummaryWriter("log_zwx")

# 训练
for step in range(total_step):
    target_features = vgg(target_img)
    content_features = vgg(content_img)
    style_features = vgg(style_img)

    stype_loss = 0
    content_loss = 0
    # 计算损失
    for param1, param2, param3 in zip(target_features, content_features, style_features):
        content_loss = torch.mean((param1 - param2) ** 2) + content_loss
        _, c, h, w = param1.size()
        param1 = param1.view(c, h * w)
        param3 = param3.view(c, h * w)

        # 算gram matrix
        param1 = torch.mm(param1, param1.t()) # .mm矩阵乘，.t()为倒置
        param3 = torch.mm(param3, param3.t())
        stype_loss = torch.mean((param1 - param3) ** 2) / (c * h * w) + stype_loss

    loss = content_loss + stype_loss * style_weight

    # 更新target_img
    optimizer.zero_grad() # 优化后梯度清0
    loss.backward() #反向传播
    optimizer.step()
    writer.add_scalar("loss_2", loss, step)
    denorm = transforms.Normalize((-2.21, -2.04, -1.80),
                                  (4.37, 4.46, 4.44))
    img = target_img.clone().squeeze()  # 降维
    img = denorm(img).clamp_(0, 1)
    img = show_img(img)
    writer.add_image("target_2", img, global_step=step)
    print("Step [{}/{}], content loss = {:.4f}, style loss = {:.4f}"
          .format(step, total_step, content_loss.item(), stype_loss.item()))

writer.close()

