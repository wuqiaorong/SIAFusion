import os
import torch
import torch.nn as nn
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
import random
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.autograd import Variable
from args_fusion import args
from imageio import imread,imsave
from PIL import Image
import cv2
import math
import matplotlib as mpl
import imageio
from torchvision import transforms
import torch.nn.functional as F


def gradient(x):
    dim = x.shape
    if (args.cuda):
        x = x.cuda(int(args.device))
    # kernel = [[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]];
    kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(dim[1], dim[1], 1, 1)
    weight = nn.Parameter(data=kernel, requires_grad=False)
    if (args.cuda):
        weight = weight.cuda(int(args.device))
    pad = nn.ReflectionPad2d(1)
    gradMap = F.conv2d(pad(x), weight=weight, stride=1, padding=0)
    # showTensor(gradMap);
    return gradMap
def load_datasetPair(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))
    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        image_path = image_path[:-mod]
    num_imgs-=mod
    original_img_path = image_path[:num_imgs]

    # random
    random.shuffle(original_img_path)
    batches = int(len(original_img_path) // BATCH_SIZE)
    return original_img_path, batches
def generateTrainNumberIndex():
    imagePatches = []
    for i in range(0+1,args.trainNumber+1):
        imagePatches.append(str(i))
    return imagePatches


#图像增强去雾
# 线性拉伸处理
# 去掉最大最小0.5%的像素值 线性拉伸至[0,1]
def stretchImage(data, s=0.005, bins=2000):
    ht = np.histogram(data, bins);
    d = np.cumsum(ht[0]) / float(data.size)
    lmin = 0;
    lmax = bins - 1
    while lmin < bins:
        if d[lmin] >= s:
            break
        lmin += 1
    while lmax >= 0:
        if d[lmax] <= 1 - s:
            break
        lmax -= 1
    return np.clip((data - ht[1][lmin]) / (ht[1][lmax] - ht[1][lmin]), 0, 1)


# 根据半径计算权重参数矩阵
g_para = {}


def getPara(radius=5):
    global g_para
    m = g_para.get(radius, None)
    if m is not None:
        return m
    size = radius * 2 + 1
    m = np.zeros((size, size))
    for h in range(-radius, radius + 1):
        for w in range(-radius, radius + 1):
            if h == 0 and w == 0:
                continue
            m[radius + h, radius + w] = 1.0 / math.sqrt(h ** 2 + w ** 2)
    m /= m.sum()
    g_para[radius] = m
    return m


# 常规的ACE实现
def zmIce(I, ratio=4, radius=300):
    para = getPara(radius)
    height, width = I.shape
    zh = []
    zw = []
    n = 0
    while n < radius:
        zh.append(0)
        zw.append(0)
        n += 1
    for n in range(height):
        zh.append(n)
    for n in range(width):
        zw.append(n)
    n = 0
    while n < radius:
        zh.append(height - 1)
        zw.append(width - 1)
        n += 1
    # print(zh)
    # print(zw)

    Z = I[np.ix_(zh, zw)]
    res = np.zeros(I.shape)
    for h in range(radius * 2 + 1):
        for w in range(radius * 2 + 1):
            if para[h][w] == 0:
                continue
            res += (para[h][w] * np.clip((I - Z[h:h + height, w:w + width]) * ratio, -1, 1))
    return res


# 单通道ACE快速增强实现
def zmIceFast(I, ratio, radius):
    # print(I.shape)
    height, width = I.shape
    if min(height, width) <= 2:
        return np.zeros(I.shape) + 0.5
    Rs = cv2.resize(I, (int((width + 1) / 2), int((height + 1) / 2)))
    Rf = zmIceFast(Rs, ratio, radius)  # 递归调用
    Rf = cv2.resize(Rf, (width, height))
    Rs = cv2.resize(Rs, (width, height))

    return Rf + zmIce(I, ratio, radius) - zmIce(Rs, ratio, radius)


# 单灰度通道加强 ratio是对比度增强因子 radius是卷积模板半径
def zmIceColor1(I, ratio=4, radius=3):
    # res = np.zeros(I.shape)
    res1= stretchImage(zmIceFast(I, ratio, radius))

    return res1
def zmMinFilterGray(src, r=5):
    '''最小值滤波，r是滤波器半径'''
    '''if r <= 0:
        return src
    h, w = src.shape[:2]
    I = src
    res = np.minimum(I  , I[[0]+range(h-1)  , :])
    res = np.minimum(res, I[range(1,h)+[h-1], :])
    I = res
    res = np.minimum(I  , I[:, [0]+range(w-1)])
    res = np.minimum(res, I[:, range(1,w)+[w-1]])
    return zmMinFilterGray(res, r-1)'''
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))  # 使用opencv的erode函数更高效


def guidedfilter(I, p, r, eps):
    '''引导滤波，直接参考网上的matlab代码'''
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def getV1(m, r, eps, w, maxV1):  # 输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = （1-t）A'''
    # V1 = np.min(m, 2)  # 得到暗通道图像
    V1=m
    # print(V1.shape)
    V1 = guidedfilter(V1, zmMinFilterGray(V1, 7), r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)  # 计算大气光照A
    # print(ht[1][1998])
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    # print(lmax)
    A = m[V1 >= ht[1][lmax]].max()

    V1 = np.minimum(V1 * w, maxV1)  # 对值范围进行限制

    return V1, A


def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    V1, A = getV1(m, r, eps, w, maxV1)  # 得到遮罩图像和大气光照
    # for k in range(3):
    Y = (m - V1) / (1 - V1 / A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校正,默认不进行该操作
    return Y

def get_T_A(m, r=81, eps=0.001, w=0.95, maxV1=0.80):
    Y = np.zeros(m.shape)
    V1, A = getV1(m, r, eps, w, maxV1)  # 得到遮罩图像和大气光照
    T=1-(V1/A)
    # for k in range(3):
    # Y = (m - V1) / (1 - V1 / A)  # 颜色校正
    # Y = np.clip(Y, 0, 1)
    # if bGamma:
    #     Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校正,默认不进行该操作
    return T,A

def get_TG(m, r=81, eps=0.001, w=0.95, maxV1=0.80):
    Y = np.zeros(m.shape)
    V1, A = getV1(m, r, eps, w, maxV1)  # 得到遮罩图像和大气光照
    T=1-(V1/A)
    T_G=cv2.Laplacian(T, cv2.CV_64F)
    # for k in range(3):
    # Y = (m - V1) / (1 - V1 / A)  # 颜色校正
    # Y = np.clip(Y, 0, 1)
    # if bGamma:
    #     Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校正,默认不进行该操作
    return T_G






def imresize(arr, size, interp='bilinear', mode=None):
    im = Image.fromarray(arr, mode=mode)
    ts = type(size)
    if np.issubdtype(ts, np.signedinteger):
        percent = size / 100.0
        size = tuple((np.array(im.size)*percent).astype(int))
    elif np.issubdtype(type(size), np.floating):
        size = tuple((np.array(im.size)*size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp])
    return np.array(imnew)

def list_images(dimgectory):
    images = []
    names = []
    dimg = listdir(dimgectory)
    dimg.sort()
    for file in dimg:
        name = file.lower()
        images.append(join(dimgectory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images


def make_floor(path1,path2):
    path = os.path.join(path1,path2)
    if os.path.exists(path) is False:
        os.makedirs(path)
    return path


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=True):
    if cuda:
        # img = tensor.clone().cpu().clamp(0, 1).numpy()
        img = tensor.cpu().clamp(0, 255).data[0].numpy()
    else:
        # img = tensor.clone().clamp(0, 1).numpy()
        img = tensor.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def matSqrt(x):
    U, D, V = torch.svd(x)
    return U * (D.pow(0.5).diag()) * V.t()


# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches


def load_dataset_aligned(image_path_ir, image_path_vi, text_mask_path,text_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = min(len(image_path_ir), len(image_path_vi), len(text_mask_path),len(text_path))

    # 截取相同数量的样本
    original_imgs_path_ir = image_path_ir[:num_imgs]
    original_imgs_path_vi = image_path_vi[:num_imgs]
    original_text_mask_path = text_mask_path[:num_imgs]
    original_text_path = text_path[:num_imgs]

    # 使用相同种子进行随机打乱，保持对应关系
    random.seed(42)  # 使用固定种子确保可重复性
    combined = list(zip(original_imgs_path_ir, original_imgs_path_vi, original_text_mask_path,original_text_path))
    random.shuffle(combined)
    original_imgs_path_ir[:], original_imgs_path_vi[:], original_text_mask_path[:], original_text_path[:] = zip(*combined)

    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path_ir = original_imgs_path_ir[:-mod]
        original_imgs_path_vi = original_imgs_path_vi[:-mod]
        original_text_mask_path = original_text_mask_path[:-mod]
        original_text_path = original_text_path[:-mod]

    batches = int(len(original_imgs_path_ir) // BATCH_SIZE)
    return original_imgs_path_ir, original_imgs_path_vi,original_text_mask_path, original_text_path, batches


def save_image_test(img_fusion, output_path):
    img_fusion = img_fusion.float()
    if args.cuda:
        img_fusion = img_fusion.cpu().data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

    # img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    # img_fusion = img_fusion * 255
    print(img_fusion.shape)
    imsave(output_path, img_fusion)


def get_image(path, height=224, width=224, flag=False):
    image = imread(path, mode='L')
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image * 255
    # image = (image-127.5)/127.5

    if height is not None and width is not None:
        image = imresize(image, [height, width], interp='nearest')
    return image

def get_image_results(path, height=224, width=224, flag=False):
    image = imread(path, mode='L')
    # image = (image - np.min(image)) / (np.max(image) - np.min(image))
    # image = image * 255
    # image = (image-127.5)/127.5

    if height is not None and width is not None:
        image = imresize(image, [height, width], interp='nearest')
    return image


def get_train_images_auto(paths, height=480, width=360, flag=False,patch_size=4):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width,  flag)
        if flag is True:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image, [1, height, width])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    images = check_image_size(images, patch_size)
    # images=images/255.
    # images = (images - torch.min(images)) / (torch.max(images) - torch.min(images))
    # images = images * 255
    # images = (images-127.5)/127.5
    return images
def get_train_images_auto_mask(paths, height=480, width=360, flag=False,patch_size=4):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width,  flag)
        if flag is True:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image, [1, height, width])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    images = check_image_size(images, patch_size)
    # images=images/255.
    # images = (images - torch.min(images)) / (torch.max(images) - torch.min(images))
    # images = images * 255
    # images = (images-127.5)/127.5
    return images/255.
def get_train_images_auto_vi(paths, height=224, width=224, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width,  flag)
        image=(image-np.min(image))/(np.max(image)-np.min(image))
        image =deHaze(image)*255
        if flag is True:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image, [1, height, width])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    images = (images - torch.min(images)) / (torch.max(images) - torch.min(images))
    images = images * 255
    # images = (images-127.5)/127.5
    return images


def check_image_size(x,patch_size):
    # NOTE: for I2I test
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x
def get_test_images_single(paths, height=None, width=None, flag=False,patch_size=8):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    c=0
    # c=0
    for path in paths:
        image = get_image(path, height, width,  flag)
        if height is not None and width is not None:

            image = imresize(image, [height, width], interp='nearest')
            image = np.array(image)  # list转numpy.array
            image = torch.from_numpy(image)  # array2tensor
            images=image
            images = torch.reshape(images, [1, 1, height, width])
            images=images.float()
            height=images.shape[2]
            width=images.shape[3]
        else:
            image = np.array(image)
            image = torch.from_numpy(image)
            images = image
            # print(image.shape)
            images = images.unsqueeze(0).unsqueeze(0)
            # print(images.shape)
            images = images.float()
            # images = check_image_size(images,patch_size)
            height = images.shape[2]
            width = images.shape[3]
            # print(image.shape)

    #     # image = imresize(image, [224, 224], interp='nearest')
    #     base_size = 224
    #     # image = torch.from_numpy(image)
    #     # images = torch.reshape(image, [1,1, 224, 224])
    #     h = image.shape[0]
    #     w = image.shape[1]
    #     if 1 * base_size < h < 2 * base_size and 1 * base_size < w < 2 * base_size:
    #         c = 4
    #         images = get_img_parts1(image, h, w)
    #     if 2*base_size<h < 3*base_size and 2*base_size< w < 3*base_size:
    #         c = 9
    #         images = get_img_parts2(image, h, w)
    #     if 2 * base_size < h < 3 * base_size and 3 * base_size < w < 4 * base_size:
    #         c = 12
    #         images = get_img_parts3(image, h, w)
    #     if 1 * base_size < h < 2 * base_size and 2 * base_size < w < 3 * base_size:
    #         c = 6
    #         images = get_img_parts4(image, h, w)
    #     if 3 * base_size < h < 4 * base_size and 4 * base_size < w < 5 * base_size:
    #         c = 20
    #         images = get_img_parts5(image, h, w)
    #     if 0 * base_size < h < 1 * base_size and 1 * base_size < w < 2 * base_size:
    #         c = 2
    #         images = get_img_parts6(image, h, w)
    #     if 0 * base_size < h < 1 * base_size and 2 * base_size < w < 3 * base_size:
    #         c = 3
    #         images = get_img_parts7(image, h, w)
    #     if h == 1 * base_size and 2 * base_size < w < 3 * base_size:
    #         c = 3
    #         images = get_img_parts8(image, h, w)
    # # c=0

    return images,height,width,c
def get_test_images(paths, height=None, width=None, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    c=0
    # c=0
    for path in paths:
        image = get_image(path, height, width,  flag)
        if height is not None and width is not None:
            image = imresize(image, [height, width], interp='nearest')
            image = np.array(image)  # list转numpy.array
            image = torch.from_numpy(image)  # array2tensor
            images=image
            images = torch.reshape(images, [1, 1, height, width])
            images=images.float()
            # print(image.shape)

        # image = imresize(image, [224, 224], interp='nearest')
        base_size = 224
        # image = torch.from_numpy(image)
        # images = torch.reshape(image, [1,1, 224, 224])
        h = image.shape[0]
        w = image.shape[1]
        if 1 * base_size < h < 2 * base_size and 1 * base_size < w < 2 * base_size:
            c = 4
            images = get_img_parts1(image, h, w)
        if 2*base_size<h < 3*base_size and 2*base_size< w < 3*base_size:
            c = 9
            images = get_img_parts2(image, h, w)
        if 2 * base_size < h < 3 * base_size and 3 * base_size < w < 4 * base_size:
            c = 12
            images = get_img_parts3(image, h, w)
        if 1 * base_size < h < 2 * base_size and 2 * base_size < w < 3 * base_size:
            c = 6
            images = get_img_parts4(image, h, w)
        if 3 * base_size < h < 4 * base_size and 4 * base_size < w < 5 * base_size:
            c = 20
            images = get_img_parts5(image, h, w)
        if 0 * base_size < h < 1 * base_size and 1 * base_size < w < 2 * base_size:
            c = 2
            images = get_img_parts6(image, h, w)
        if 0 * base_size < h < 1 * base_size and 2 * base_size < w < 3 * base_size:
            c = 3
            images = get_img_parts7(image, h, w)
        if h == 1 * base_size and 2 * base_size < w < 3 * base_size:
            c = 3
            images = get_img_parts8(image, h, w)
    # c=0

    return images, h, w, c
def get_test_images_dense(paths, height=None, width=None, mode='L'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            # test = ImageToTensor(image).numpy()
            # shape = ImageToTensor(image).size()
            image = ImageToTensor(image).float().numpy()*255
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images
def get_test_images_vi(paths, fog_model,height=None, width=None, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    c = 0
    # c=0
    for path in paths:
        image = get_image(path, height, width,  flag)
        transrorm = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                    torchvision.transforms.ToTensor(),
                                                    ])
        # # image=zmIceColor1(image/255)*255
        # # image = deHaze(image/255)*255
        # fog_model = Defogging()
        # fog_model.load_state_dict(torch.load(path_fog))
        # image1=(image-np.min(image))/(np.max(image)-np.min(image))
        image1=Image.fromarray(image)
        image1=transrorm(image1)
        # # print(image1.shape)
        image1 = torch.reshape(image1, (1, 1,224, 224))
        # image1 = image1
        # print(image1.shape)

        image1=image1.cuda()
        # image2=model_fusion.img_dazing()
        # print(image1.shape)
        # print(fog_model.shape)
        # image1 = (image1 - torch.min(image1)) / (torch.max(image1) - torch.min(image1))
        Fb = fog_model(image1/255.0)
        # image=image.cuda()
        my_softmax = nn.Softmax(dim=1)
        probs = my_softmax(Fb)
        w1 = probs[0][0].item()
        w2 = probs[0][1].item()
        w_vi = w1
        w_ir = w2
        print(w1)
        print(w2)
        # image=image.cpu()
        # image_j =deHaze(image/255)*255
        image_j =zmIceColor1(image/255)*255

        # image= w_vi * image + w_ir * image_j
        # image = (image - np.min(image)) / (np.max(image) - np.min(image))
        # image = image * 255

        image = image_j
        if height is not None and width is not None:
            image = imresize(image, [height, width], interp='nearest')
        # image = imresize(image, [224, 224], interp='nearest')
        base_size = 224
        # image = torch.from_numpy(image)
        # images = torch.reshape(image, [1,1, 224, 224])
        h = image.shape[0]
        w = image.shape[1]
        if 1 * base_size < h < 2 * base_size and 1 * base_size < w < 2 * base_size:
            c = 4
            images = get_img_parts1(image, h, w)
        if 2*base_size<h < 3*base_size and 2*base_size< w < 3*base_size:
            c = 9
            images = get_img_parts2(image, h, w)
        if 2 * base_size < h < 3 * base_size and 3 * base_size < w < 4 * base_size:
            c = 12
            images = get_img_parts3(image, h, w)
        if 1 * base_size < h < 2 * base_size and 2 * base_size < w < 3 * base_size:
            c = 6
            images = get_img_parts4(image, h, w)
        if 3 * base_size < h < 4 * base_size and 4 * base_size < w < 5 * base_size:
            c = 20
            images = get_img_parts5(image, h, w)
        if 0 * base_size < h < 1 * base_size and 1 * base_size < w < 2 * base_size:
            c = 2
            images = get_img_parts6(image, h, w)
        if 0 * base_size < h < 1 * base_size and 2 * base_size < w < 3 * base_size:
            c = 3
            images = get_img_parts7(image, h, w)
        if h == 1 * base_size and 2 * base_size < w < 3 * base_size:
            c = 3
            images = get_img_parts8(image, h, w)
    # c=0

    return images, h, w, c
def get_test_images_vi2(paths, paths2,model_fusion,height=None, width=None, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    c = 0
    # c=0
    for path in paths:
        image_vi = get_image(path, height, width,  flag)
        image_ir=get_image(paths2,height, width,  flag)
        transrorm = torchvision.transforms.Compose([#torchvision.transforms.Resize((224, 224)),
                                                    torchvision.transforms.ToTensor(),
                                                    ])
        # # image=zmIceColor1(image/255)*255
        # # image = deHaze(image/255)*255
        # fog_model = Defogging()
        # fog_model.load_state_dict(torch.load(path_fog))
        # image1=(image-np.min(image))/(np.max(image)-np.min(image))
        image1=Image.fromarray(image_vi)
        image1=transrorm(image1)
        # # print(image1.shape)
        image1 = torch.reshape(image1, (1, 1,image1.shape[1], image1.shape[2]))
        image2=Image.fromarray(image_ir)
        image2= transrorm(image2)
        image2 = torch.reshape(image2, (1, 1, image2.shape[1], image2.shape[2]))
        # image1 = image1
        # print(image1.shape)

        image1=image1.cuda()
        image2=image2.cuda()
        image3=model_fusion.img_dazing(image1,image2)
        # print(image1.shape)
        # print(fog_model.shape)
        # image1 = (image1 - torch.min(image1)) / (torch.max(image1) - torch.min(image1))
        # Fb = fog_model(image1/255.0)
        # # image=image.cuda()
        # my_softmax = nn.Softmax(dim=1)
        # probs = my_softmax(Fb)
        # w1 = probs[0][0].item()
        # w2 = probs[0][1].item()
        # w_vi = w1
        # w_ir = w2
        # print(w1)
        # print(w2)
        # # image=image.cpu()
        # # image_j =deHaze(image/255)*255
        # image_j =zmIceColor1(image/255)*255

        # image= w_vi * image + w_ir * image_j
        # image = (image - np.min(image)) / (np.max(image) - np.min(image))
        # image = image * 255

        # toPIL = transforms.ToPILImage()
        # pic=image3.detach().cpu().numpy()[0][0]
        # print(pic.shape)
        # pic.save("/root/autodl-tmp/swim_new/image_defog/1.png")
        image = image3.cpu()
        image = image.reshape(image.shape[2], image.shape[3])
        image=np.array(image)
        # print(image)
        # cv2.imwrite("/root/autodl-tmp/swim_new/image_defog/1.png",pic)
        # image.save("/root/autodl-tmp/swim_new/image_defog/1.png")

        # print(image.shape[2])

        if height is not None and width is not None:
            image = imresize(image, [height, width], interp='nearest')
        # image = imresize(image, [224, 224], interp='nearest')
        base_size = 224
        # image = torch.from_numpy(image)
        # images = torch.reshape(image, [1,1, 224, 224])
        h = image.shape[0]
        w = image.shape[1]
        if 1 * base_size < h < 2 * base_size and 1 * base_size < w < 2 * base_size:
            c = 4
            images = get_img_parts1(image, h, w)
        if 2*base_size<h < 3*base_size and 2*base_size< w < 3*base_size:
            c = 9
            images = get_img_parts2(image, h, w)
        if 2 * base_size < h < 3 * base_size and 3 * base_size < w < 4 * base_size:
            c = 12
            images = get_img_parts3(image, h, w)
        if 1 * base_size < h < 2 * base_size and 2 * base_size < w < 3 * base_size:
            c = 6
            images = get_img_parts4(image, h, w)
        if 3 * base_size < h < 4 * base_size and 4 * base_size < w < 5 * base_size:
            c = 20
            images = get_img_parts5(image, h, w)
        if 0 * base_size < h < 1 * base_size and 1 * base_size < w < 2 * base_size:
            c = 2
            images = get_img_parts6(image, h, w)
        if 0 * base_size < h < 1 * base_size and 2 * base_size < w < 3 * base_size:
            c = 3
            images = get_img_parts7(image, h, w)
        if h == 1 * base_size and 2 * base_size < w < 3 * base_size:
            c = 3
            images = get_img_parts8(image, h, w)
    # c=0

    return images, h, w, c



def get_img_parts1(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 448-w, 0, 448-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:224, 0: 224]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:224, 224: 448]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[224:448, 0: 224]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[224:448, 224: 448]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    return images


def get_img_parts2(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 672-w, 0, 672-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:224, 0: 224]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:224, 224: 448]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:224, 448: 672]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[224:448, 0: 224]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[224:448, 224: 448]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[224:448, 448: 672]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    img7 = image[448:672, 0: 224]
    img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
    img8 = image[448:672, 224: 448]
    img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])
    img9 = image[448:672, 448: 672]
    img9 = torch.reshape(img9, [1, 1, img9.shape[0], img9.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    images.append(img7.float())
    images.append(img8.float())
    images.append(img9.float())
    return images


def get_img_parts3(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 896-w, 0, 672-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:224, 0: 224]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:224, 224: 448]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:224, 448: 672]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[0:224, 672: 896]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[224:448, 0: 224]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[224:448, 224: 448]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    img7 = image[224:448, 448: 672]
    img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
    img8 = image[224:448, 672: 896]
    img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])
    img9 = image[448:672, 0: 224]
    img9 = torch.reshape(img9, [1, 1, img9.shape[0], img9.shape[1]])
    img10 = image[448:672, 224: 448]
    img10 = torch.reshape(img10, [1, 1, img10.shape[0], img10.shape[1]])
    img11 = image[448:672, 448: 672]
    img11 = torch.reshape(img11, [1, 1, img11.shape[0], img11.shape[1]])
    img12 = image[448:672, 672: 896]
    img12 = torch.reshape(img12, [1, 1, img12.shape[0], img12.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    images.append(img7.float())
    images.append(img8.float())
    images.append(img9.float())
    images.append(img10.float())
    images.append(img11.float())
    images.append(img12.float())
    return images


def get_img_parts4(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 672-w, 0, 448-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:224, 0: 224]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:224, 224: 448]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:224, 448: 672]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[224:448, 0: 224]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[224:448, 224: 448]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[224:448, 448: 672]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    return images


def get_img_parts5(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 1120-w, 0, 896-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:224, 0: 224]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:224, 224: 448]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:224, 448: 672]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[0:224, 672: 896]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[0:224, 896: 1120]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[224:448, 0: 224]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    img7 = image[224:448, 224: 448]
    img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
    img8 = image[224:448, 448: 672]
    img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])
    img9 = image[224:448, 672: 896]
    img9 = torch.reshape(img9, [1, 1, img9.shape[0], img9.shape[1]])
    img10 = image[224:448, 896: 1120]
    img10 = torch.reshape(img10, [1, 1, img10.shape[0], img10.shape[1]])
    img11 = image[448:672, 0: 224]
    img11 = torch.reshape(img11, [1, 1, img11.shape[0], img11.shape[1]])
    img12 = image[448:672, 224: 448]
    img12 = torch.reshape(img12, [1, 1, img12.shape[0], img12.shape[1]])
    img13 = image[448:672, 448: 672]
    img13 = torch.reshape(img13, [1, 1, img13.shape[0], img13.shape[1]])
    img14 = image[448:672, 672: 896]
    img14 = torch.reshape(img14, [1, 1, img14.shape[0], img14.shape[1]])
    img15 = image[448:672, 896: 1120]
    img15 = torch.reshape(img15, [1, 1, img15.shape[0], img15.shape[1]])
    img16 = image[672:896, 0: 224]
    img16 = torch.reshape(img16, [1, 1, img16.shape[0], img16.shape[1]])
    img17 = image[672:896, 224: 448]
    img17 = torch.reshape(img17, [1, 1, img17.shape[0], img17.shape[1]])
    img18 = image[672:896, 448: 672]
    img18 = torch.reshape(img18, [1, 1, img18.shape[0], img18.shape[1]])
    img19 = image[672:896, 672: 896]
    img19 = torch.reshape(img19, [1, 1, img19.shape[0], img19.shape[1]])
    img20 = image[672:896, 896: 1120]
    img20 = torch.reshape(img20, [1, 1, img20.shape[0], img20.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    images.append(img7.float())
    images.append(img8.float())
    images.append(img9.float())
    images.append(img10.float())
    images.append(img11.float())
    images.append(img12.float())
    images.append(img13.float())
    images.append(img14.float())
    images.append(img15.float())
    images.append(img16.float())
    images.append(img17.float())
    images.append(img18.float())
    images.append(img19.float())
    images.append(img20.float())

    return images


def get_img_parts6(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 448-w, 0, 224-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:224, 0: 224]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:224, 224: 448]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    return images


def get_img_parts7(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 672-w, 0, 224-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:224, 0: 224]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:224, 224: 448]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:224, 448: 672]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    return images


def get_img_parts8(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 672-w, 0, 224), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:224, 0: 224]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:224, 224: 448]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:224, 448: 672]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    return images


def recons_fusion_images1(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]

        img_f = torch.zeros(1, h, w).cuda()
        print(img_f.size())

        img_f[:, 0:224, 0: 224] += img1
        img_f[:, 0:224, 224: w] += img2[:, 0:224, 0:w-224]
        img_f[:, 224:h, 0: 224] += img3[:, 0:h-224, 0:224]
        img_f[:, 224:h, 224: w] += img4[:, 0:h-224, 0:w-224]
        # img_f=(img_f - torch.min(img_f)) / (torch.max(img_f) - torch.min(img_f))

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images2(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]
        img7 = img_lists[6][i]
        img8 = img_lists[7][i]
        img9 = img_lists[8][i]
        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:224, 0: 224] += img1
        img_f[:, 0:224, 224: 448] += img2
        img_f[:, 0:224, 448: w] += img3[:, 0:224, 0:w-448]
        img_f[:, 224:448, 0: 224] += img4
        img_f[:, 224:448, 224: 448] += img5
        img_f[:, 224:448, 448: w] += img6[:, 0:224, 0:w-448]
        img_f[:, 448:h, 0: 224] += img7[:, 0:h-448, 0:224]
        img_f[:, 448:h, 224: 448] += img8[:, 0:h-448, 0:224]
        img_f[:, 448:h, 448: w] += img9[:, 0:h-448, 0:w - 448]
        # img_f = (img_f - torch.min(img_f)) / (torch.max(img_f) - torch.min(img_f))
        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images3(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]
        img7 = img_lists[6][i]
        img8 = img_lists[7][i]
        img9 = img_lists[8][i]
        img10 = img_lists[9][i]
        img11 = img_lists[10][i]
        img12 = img_lists[11][i]
        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:224, 0: 224] += img1
        img_f[:, 0:224, 224: 448] += img2
        img_f[:, 0:224, 448: 672] += img3
        img_f[:, 0:224, 672: w] += img4[:, 0:224, 0:w-672]
        img_f[:, 224:448, 0: 224] += img5
        img_f[:, 224:448, 224: 448] += img6
        img_f[:, 224:448, 448: 672] += img7
        img_f[:, 224:448, 672: w] += img8[:, 0:224, 0:w-672]
        img_f[:, 448:h, 0: 224] += img9[:, 0:h-448, 0:224]
        img_f[:, 448:h, 224: 448] += img10[:, 0:h - 448, 0:224]
        img_f[:, 448:h, 448: 672] += img11[:, 0:h - 448, 0:224]
        img_f[:, 448:h, 672: w] += img12[:, 0:h - 448, 0:w-672]
        # img_f = (img_f - torch.min(img_f)) / (torch.max(img_f) - torch.min(img_f))

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images4(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]

        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:224, 0: 224] += img1
        img_f[:, 0:224, 224: 448] += img2
        img_f[:, 0:224, 448: w] += img3[:, 0:224, 0:w-448]
        img_f[:, 224:h, 0: 224] += img4[:, 0:h-224, 0:224]
        img_f[:, 224:h, 224: 448] += img5[:, 0:h - 224, 0:224]
        img_f[:, 224:h, 448: w] += img6[:, 0:h - 224, 0:w-448]
        # img_f = (img_f - torch.min(img_f)) / (torch.max(img_f) - torch.min(img_f))

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images5(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]
        img7 = img_lists[6][i]
        img8 = img_lists[7][i]
        img9 = img_lists[8][i]
        img10 = img_lists[9][i]
        img11 = img_lists[10][i]
        img12 = img_lists[11][i]
        img13 = img_lists[12][i]
        img14 = img_lists[13][i]
        img15 = img_lists[14][i]
        img16 = img_lists[15][i]
        img17 = img_lists[16][i]
        img18 = img_lists[17][i]
        img19 = img_lists[18][i]
        img20 = img_lists[19][i]
        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:224, 0: 224] += img1
        img_f[:, 0:224, 224: 448] += img2
        img_f[:, 0:224, 448: 672] += img3
        img_f[:, 0:224, 672: 896] += img4
        img_f[:, 0:224, 896: w] += img5[:, 0:224, 0:w-896]
        img_f[:, 224:448, 0: 224] += img6
        img_f[:, 224:448, 224: 448] += img7
        img_f[:, 224:448, 448: 672] += img8
        img_f[:, 224:448, 672: 896] += img9
        img_f[:, 224:448, 896: w] += img10[:, 0:224, 0:w-896]
        img_f[:, 448:672, 0: 224] += img11
        img_f[:, 448:672, 224: 448] += img12
        img_f[:, 448:672, 448: 672] += img13
        img_f[:, 448:672, 672: 896] += img14
        img_f[:, 448:672, 896: w] += img15[:, 0:224, 0:w - 896]
        img_f[:, 672:h, 0: 224] += img16[:, 0:h-672, 0:224]
        img_f[:, 672:h, 224: 448] += img17[:, 0:h-672, 0:224]
        img_f[:, 672:h, 448: 672] += img18[:, 0:h-672, 0:224]
        img_f[:, 672:h, 672: 896] += img19[:, 0:h-672, 0:224]
        img_f[:, 672:h, 896: w] += img20[:, 0:h-672, 0:w - 896]
        # img_f = (img_f - torch.min(img_f)) / (torch.max(img_f) - torch.min(img_f))

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images6(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]

        img_f = torch.zeros(1, h, w).cuda()
        print(img_f.size())

        img_f[:, 0:h, 0: 224] += img1[:, 0:h, 0:224]
        img_f[:, 0:h, 224: w] += img2[:, 0:h, 0:w-224]
        # img_f = (img_f - torch.min(img_f)) / (torch.max(img_f) - torch.min(img_f))

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images7(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]

        img_f = torch.zeros(1, h, w).cuda()
        print(img_f.size())

        img_f[:, 0:h, 0: 224] += img1[:, 0:h, 0:224]
        img_f[:, 0:h, 224: 448] += img2[:, 0:h, 0:224]
        img_f[:, 0:h, 448: w] += img3[:, 0:h, 0:w - 448]
        # img_f = (img_f - torch.min(img_f)) / (torch.max(img_f) - torch.min(img_f))

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images8(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]

        img_f = torch.zeros(1, h, w).cuda()
        print(img_f.size())

        img_f[:, 0:h, 0: 224] += img1[:, 0:h, 0:224]
        img_f[:, 0:h, 224: 448] += img2[:, 0:h, 0:224]
        img_f[:, 0:h, 448: w] += img3[:, 0:h, 0:w - 448]
        # img_f = (img_f - torch.min(img_f)) / (torch.max(img_f) - torch.min(img_f))

        img_f_list.append(img_f)
    return img_f_list

# def remove_noise_gaussian(image, kernel_size=(5, 5), sigmaX=0):
#     return cv2.GaussianBlur(image, kernel_size, sigmaX)

def save_image_test(img_fusion, output_path):
    img_fusion = img_fusion.float()
    # print(img_fusion)
    if args.cuda:
        img_fusion = img_fusion.cpu().data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()
    # img_fusion = ((img_fusion / 2) + 0.5) * 255
    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    # img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    # # img_fusion =deHaze(img_fusion)
    # # img_fusion =zmIceColor1(img_fusion)
    img_fusion = img_fusion * 255
    # img_fusion = remove_noise_gaussian(img_fusion)  # Apply Gaussian filter
    print(img_fusion.shape)
    img_fusion = img_fusion.reshape([1,img_fusion.shape[0], img_fusion.shape[1]])#有些出来的是二维，需要变成3维
    # img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    if img_fusion.shape[2] == 1:
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    imageio.imwrite(output_path, img_fusion)

def save_image_test_resize(img_fusion, output_path,h,w):
    img_fusion = img_fusion.float()
    # print(img_fusion)
    if args.cuda:
        img_fusion = img_fusion.cpu().data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()
    # img_fusion = ((img_fusion / 2) + 0.5) * 255
    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    # img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    # # img_fusion =deHaze(img_fusion)
    # # img_fusion =zmIceColor1(img_fusion)
    img_fusion = img_fusion * 255
    # img_fusion = remove_noise_gaussian(img_fusion)  # Apply Gaussian filter

    # img_fusion = img_fusion.reshape([1,img_fusion.shape[0], img_fusion.shape[1]])#有些出来的是二维，需要变成3维
    img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    if img_fusion.shape[2] == 1:
        img_fusion = cv2.resize(img_fusion, (w, h), interpolation=cv2.INTER_LINEAR)
    print(img_fusion.shape)
    imageio.imwrite(output_path, img_fusion)
def save_image_test_dense(img_fusion, output_path):
    img_fusion = img_fusion.float()
    # print(img_fusion)
    if args.cuda:
        img_fusion = img_fusion.cpu().clamp(0, 255).data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()
    # img_fusion = ((img_fusion / 2) + 0.5) * 255
    # img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    # # img_fusion =deHaze(img_fusion)
    # # img_fusion =zmIceColor1(img_fusion)
    # img_fusion = img_fusion * 255
    print(img_fusion.shape)
    # img_fusion = img_fusion.reshape([1,img_fusion.shape[0], img_fusion.shape[1]])#有些出来的是二维，需要变成3维
    img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    # if img_fusion.shape[2] == 1:
    #     img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    save_imgs(output_path, img_fusion)


def save_imgs(path, img_fusion):
    if img_fusion.shape[2] == 1:
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    imsave(path, img_fusion)


def save_image_scales(img_fusion, output_path):
    img_fusion = img_fusion.float()
    img_fusion = img_fusion.cpu().data[0].numpy()
    imageio.imwrite(output_path, img_fusion)


