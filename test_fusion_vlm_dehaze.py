# test phase
import os
import torch
from torch.autograd import Variable
# from net_patch1 import SwinFuse,Fusion_network
from netvlm_defog4 import SwinFuse, FusionBlock_res
import utils
from args_fusion import args
import numpy as np
import fix_feature_get
import time
import cv2
from mask_genrate_1 import load_model_mask,load_and_preprocess_image,generate_mask_from_prompts,combine_masks
import clip
import matplotlib.pyplot as plt
def generate_mask(model,infrared_image_path,visible_image_path,prompts_file_path):

    infrared_img = load_and_preprocess_image(infrared_image_path)
    visible_img = load_and_preprocess_image(visible_image_path)

    infrared_mask, infrared_text_feature = generate_mask_from_prompts(model, infrared_img, prompts_file_path)
    visible_mask, visible_text_feature = generate_mask_from_prompts(model, visible_img, prompts_file_path)

    combined_mask = combine_masks(infrared_mask, visible_mask)
    combined_mask = torch.from_numpy(combined_mask).float().cuda()
    return combined_mask,visible_text_feature




def load_model(path, path_fusion, in_chans, out_chans):
    nb_filter=128
    SwinFuse_model = SwinFuse(in_chans=in_chans, out_chans=out_chans,img_size=416)
    SwinFuse_model.load_state_dict(torch.load(path))
    # fusion_model = Fusion_network(nC=nb_filter)
    fusion_model = FusionBlock_res(in_chans=in_chans, out_chans=out_chans)
    fusion_model.load_state_dict(torch.load(path_fusion))
    para = sum([np.prod(list(p.size())) for p in SwinFuse_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(SwinFuse_model._get_name(), para * type_size / 1000 / 1000))

    para = sum([np.prod(list(p.size())) for p in fusion_model.parameters()])
    type_size = 1
    print('Model {} : params: {:4f}M'.format(fusion_model._get_name(), para * type_size / 1000 / 1000))


    SwinFuse_model.eval()
    fusion_model.eval()
    SwinFuse_model.cuda()
    fusion_model.cuda()

    return SwinFuse_model, fusion_model
def generate_heatmap(cam, cmap='coolwarm'):
    """Generate heatmap from CAM using a colormap"""
    cam_np = cam.cpu().numpy()[0]
    # cam_np=[cam_np,cam_np,cam_np]
    heatmap = plt.get_cmap(cmap)(cam_np)
    # print(heatmap.shape)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
    return heatmap
import matplotlib.pyplot as plt
import os
import math


# 计算特征图的类激活映射并保存
def compute_and_save_cam_features(features, output_dir, image_name_prefix):
    """
    计算特征图的类激活映射并保存
    features: tensor of shape [B, C, H, W]
    output_dir: 保存CAM图像的目录
    image_name_prefix: 图像文件名前缀
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 全局平均池化获取通道权重
    weights = torch.mean(features, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

    # 计算加权特征图
    cam = torch.sum(features * weights, dim=1, keepdim=True)  # [B, 1, H, W]

    # 激活函数
    cam = torch.relu(cam)

    # 归一化到0-1范围
    cam = (cam - torch.min(cam)) / (torch.max(cam) - torch.min(cam) + 1e-8)

    # 转换为numpy
    cam_np = cam[0, 0].cpu().detach().numpy()

    # 应用红到蓝的颜色映射
    # 将值缩放到0-255范围
    cam_normalized = (cam_np * 255).astype(np.uint8)

    # 使用OpenCV应用红到蓝的颜色映射
    colored_cam = cv2.applyColorMap(cam_normalized, cv2.COLORMAP_JET)

    # OpenCV的JET colormap是从蓝到红，我们需要反转颜色通道以获得红到蓝效果
    colored_cam = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)

    # 保存彩色图像
    output_path = os.path.join(output_dir, f"{image_name_prefix}_colored.png")
    cv2.imwrite(output_path, cv2.cvtColor(colored_cam, cv2.COLOR_RGB2BGR))

    # # 同时保存灰度图像
    # output_path_gray = os.path.join(output_dir, f"{image_name_prefix}.png")
    # cv2.imwrite(output_path_gray, cam_normalized)

    return cam


def save_feature_maps(feature, output_dir, prefix="feature", cols=8):
    """
    feature: tensor of shape [B, C, H, W]
    output_dir: 保存图像的目录
    prefix: 文件名前缀
    cols: 每行显示多少个通道
    """
    # B, C, H, W = feature.shape
    C=1
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rows = math.ceil(C / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    for c in range(C):
        row = c // cols
        col = c % cols
        ax = axes[row, col] if rows > 1 else axes[col] if cols > 1 else axes
        feature_map = feature[0].cpu().detach().numpy()
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        ax.imshow(feature_map, cmap='jet')
        ax.axis('off')

    # 关闭多余的子图
    for c in range(C, rows * cols):
        row = c // cols
        col = c % cols
        ax = axes[row, col] if rows > 1 else axes[col] if cols > 1 else axes
        ax.axis('off')

    plt.suptitle(f"{prefix} - Total {C} channels", fontsize=12)
    plt.savefig(os.path.join(output_dir, f"{prefix}_all_channels.png"), bbox_inches='tight', dpi=200)
    plt.close()


def normalize_attention_distribution(sim_features, output_dir, prefix="normalized_sim", index=None):
    """
    对特征图进行归一化处理，使总的关注度分布更加真实
    sim_features: tensor of shape [B, C, H, W]
    output_dir: 保存归一化特征图的目录
    prefix: 文件名前缀
    index: 图像索引
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取特征图
    feature_map = sim_features[0].cpu().detach().numpy()  # [C, H, W]
    # print(feature_map.shape)

    # 方法1: L1归一化 - 使所有位置的总和为1
    # 先将所有通道合并
    # combined_map = np.sum(feature_map, axis=0)  # [H, W]

    # L1归一化
    l1_norm_map = feature_map / (np.sum(feature_map) + 1e-8)

    # 保存L1归一化的结果（红蓝色映射）
    # l1_output_path = os.path.join(output_dir, f"{prefix}_l1_norm_{index}.png")
    l1_visual = (l1_norm_map - l1_norm_map.min()) / (l1_norm_map.max() - l1_norm_map.min() + 1e-8)
    l1_visual = (l1_visual * 255).astype(np.uint8)

    # 应用红到蓝的颜色映射
    l1_colored = cv2.applyColorMap(l1_visual, cv2.COLORMAP_JET)
    l1_colored = cv2.cvtColor(l1_colored, cv2.COLOR_BGR2RGB)
    l1_output_path_colored = os.path.join(output_dir, f"{prefix}_l1_norm_{index}.png")
    cv2.imwrite(l1_output_path_colored, cv2.cvtColor(l1_colored, cv2.COLOR_RGB2BGR))

    # # 保存灰度图像
    # cv2.imwrite(l1_output_path, l1_visual)

    # 方法2: Softmax归一化 - 强调相对重要性
    flattened = feature_map.flatten()
    softmax_flat = np.exp(flattened) / (np.sum(np.exp(flattened)) + 1e-8)
    softmax_map = softmax_flat.reshape(feature_map.shape)

    # 保存Softmax归一化的结果（红蓝色映射）
    # softmax_output_path = os.path.join(output_dir, f"{prefix}_softmax_norm_{index}.png")
    softmax_visual = (softmax_map - softmax_map.min()) / (softmax_map.max() - softmax_map.min() + 1e-8)
    softmax_visual = (softmax_visual * 255).astype(np.uint8)

    # 应用红到蓝的颜色映射
    softmax_colored = cv2.applyColorMap(softmax_visual, cv2.COLORMAP_JET)
    softmax_colored = cv2.cvtColor(softmax_colored, cv2.COLOR_BGR2RGB)
    softmax_output_path_colored = os.path.join(output_dir, f"{prefix}_softmax_norm_{index}.png")
    cv2.imwrite(softmax_output_path_colored, cv2.cvtColor(softmax_colored, cv2.COLOR_RGB2BGR))

    # # 保存灰度图像
    # cv2.imwrite(softmax_output_path, softmax_visual)

    return l1_norm_map, softmax_map


def save_image(image, path):
    """Save the image to the specified path"""
    cv2.imwrite(path, image)
def run_demo(SwinFuse_model,fusion_model,infrared_path, visible_path, output_path_root, index,description_feature,visual):
    img_ir,h,w,c = utils.get_test_images_single(infrared_path,height=args.height, width=args.width)
    img_vi,h,w,c = utils.get_test_images_single(visible_path,height=args.height, width=args.width)
    _, h1, w1, c1 = utils.get_test_images_single(visible_path, height=None, width=None)

    if args.cuda:
        img_ir = img_ir.cuda()
        img_vi = img_vi.cuda()
        description_feature=description_feature.cuda().float()
    img_ir = Variable(img_ir, requires_grad=False)
    img_vi = Variable(img_vi, requires_grad=False)
    description_feature=Variable(description_feature, requires_grad=False)
    # #原始模型
    # # # encoder
    # tir3_1 = SwinFuse_model.self_embed(img_ir)
    # tvi3_1 = SwinFuse_model.self_embed(img_vi)
    # tir3 = SwinFuse_model.encoder(tir3_1,h,w)
    # tvi3 = SwinFuse_model.encoder(tvi3_1,h,w)
    # # fusion
    # f = fusion_model(tir3, tvi3, description_feature)
    # # decoder
    # img_fusion = SwinFuse_model.up_x4(f)


    # #net增加关联模块
    # # encoder
    # en_vi_1 = SwinFuse_model.self_embed(img_vi, h, w)
    # en_ir_1 = SwinFuse_model.self_embed(img_ir, h, w)
    # img1, pos1 = fusion_model.pos_embed(en_vi_1, img_vi, description_feature, h, w)
    # img2, pos2 = fusion_model.pos_embed(en_ir_1, img_ir, description_feature, h, w)
    # img1_en1 = SwinFuse_model.encoder1(img1)
    # img1_en1 = fusion_model.encoder_text(img1_en1, pos1)
    # img1_en2 = SwinFuse_model.encoder1(img1_en1)
    # img1_en2 = fusion_model.encoder_text(img1_en2, pos1)
    #
    # img2_en1 = SwinFuse_model.encoder1(img2)
    # img2_en1 = fusion_model.encoder_text(img2_en1, pos2)
    # img2_en2 = SwinFuse_model.encoder1(img2_en1)
    # img2_en2 = fusion_model.encoder_text(img2_en2, pos2)
    # image_vi_pre = fusion_model.img_dazing(img_vi, img_ir)
    img1_en1, pos1 = SwinFuse_model.self_embed(img_vi, h, w)
    img2_en1, pos2 = SwinFuse_model.self_embed(img_ir, h, w)
    img1_en1_daze=fusion_model.img_dazing(img1_en1, img2_en1)

    img1_en2 = SwinFuse_model.encoder(img1_en1_daze)
    if visual:
        # 计算并保存 img1_en1_daze 的 CAM
        cam_daze_dir = os.path.join(output_path_root, "cam_img1_en1_daze")
        compute_and_save_cam_features(img1_en1_daze, cam_daze_dir, f"cam_img1_en1_daze_{index}")

        # 计算并保存 img1_en2 的 CAM
        cam_en2_dir = os.path.join(output_path_root, "cam_img1_en2")
        compute_and_save_cam_features(img1_en2, cam_en2_dir, f"cam_img1_en2_{index}")

    # sim_map = torch.einsum('bchw,blc->blhw', img1_en2, description_feature)
    # sim_map = sim_map.mean(dim=1)
    # cam,sim,cos=sim_map.softmax(dim=-1), sim_map.softmax(dim=-1), description_feature
    # img1_texture, sim_vi, fea_cos_vi, cam_vi =cam,sim,cos,cam
    img1_texture, sim_vi, fea_cos_vi,cam_vi = fusion_model.encoder_text(img1_en2, description_feature)

    # img2_en1, pos2 = SwinFuse_model.self_embed(img_ir, h, w)
    img2_en2 = SwinFuse_model.encoder(img2_en1)
    # sim_map = torch.einsum('bchw,blc->blhw', img2_en2, description_feature)
    # sim_map = sim_map.mean(dim=1)
    # cam, sim, cos = sim_map.softmax(dim=-1), sim_map.softmax(dim=-1), description_feature
    # img2_texture,  sim_ir, fea_cos_ir,cam_ir = cam, sim, cos, cam
    img2_texture, sim_ir, fea_cos_ir,cam_ir = fusion_model.encoder_text(img2_en2, description_feature)
    # fusion
    f = fusion_model(img2_en2, img1_en2, pos_vi=img1_texture, pos_ir=img2_texture)
    # f = fusion_model(img2_en2, img1_en2, description_feature)
    # decoder
    img_fusion = SwinFuse_model.up_x4(f)
    img_fusion_list = [img_fusion]
    ############################ multi outputs ##############################################
    output_count = 0
    for img_fusion in img_fusion_list:
        # file_name = 'fusion' + '_' + str(index) + '_swinfuse_' + str(output_count) + '_' + 'f_type' + '.png'
        file_name = str(index) + '.png'
        output_path = output_path_root + file_name
        output_count += 1
        # save images
        utils.save_image_test_resize(img_fusion, output_path,h1, w1)
        if visual:
            # Generate heatmap
            heatmap_vi = generate_heatmap(cam_vi, cmap='jet')
            heatmap_ir = generate_heatmap(cam_ir, cmap='jet')
            # print(heatmap_vi.shape)
            # 假设 output_path_root 是输出根目录，index 是当前图像索引
            output_dir_vi = os.path.join(output_path_root, "sim_vi")
            output_dir_ir = os.path.join(output_path_root, "sim_ir")
            # # Save the heatmap images
            vi_heatmap_path = output_dir_vi+'/' + f'heatmap_vi_{index}.png'
            ir_heatmap_path = output_dir_ir+'/' + f'heatmap_ir_{index}.png'
            save_image(heatmap_vi, vi_heatmap_path)
            save_image(heatmap_ir, ir_heatmap_path)
            # # 假设 output_path_root 是输出根目录，index 是当前图像索引
            # output_dir_vi = os.path.join(output_path_root, "sim_vi")
            # output_dir_ir = os.path.join(output_path_root, "sim_ir")
            # save_feature_maps(sim_vi, output_dir_vi, prefix=f"sim_vi_img{index}")
            # save_feature_maps(sim_ir, output_dir_ir, prefix=f"sim_ir_img{index}")

            # 添加归一化注意力分布处理
            # 处理sim_vi
            normalized_vi_dir = os.path.join(output_path_root, "normalized_sim_vi")
            normalize_attention_distribution(sim_vi, normalized_vi_dir, "normalized_sim_vi", index)

            # 处理sim_ir
            normalized_ir_dir = os.path.join(output_path_root, "normalized_sim_ir")
            normalize_attention_distribution(sim_ir, normalized_ir_dir, "normalized_sim_ir", index)

        # print(output_path)
        # quality = fix_feature_get.show_Quality_evaluation(infrared_path, visible_path, output_path,h1,w1)
        # return quality,h,w


def main():
    visual = True
    # run demo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(
        "/media/ExtHDD/pretrained_weights/ViT-B-32.pt"
        , device=device)
    clip_model.eval()
    # test_path = "images/21_pairs_tno/"
    # test_path = "images/test01/"
    # test_path = "images/roadsense/"
    # test_path="images/defog/"
    network_type = 'SwinFuse'
    fusion_type = ['l1_mean']
    # output_path = './outputs/'
    output_path = './outputs/'
    # in_c = 3 for RGB imgs; in_c = 1 for gray imgs
    in_chans = 1

    num_classes = in_chans
    mode = 'L'
    model_path = args.model_path_gray
    modelfusion_path=args.model_fusion

    with torch.no_grad():
        print('SSIM weight ----- ' + args.ssim_path[1])
        ssim_weight_str = args.ssim_path[3]
        f_type = fusion_type[0]


        model, fusion_model = load_model(model_path,modelfusion_path, in_chans, num_classes)
        begin = time.time()
        # for a in range(10):
        num=25
        total_quality = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        total_ir_quality = [0, 0, 0, 0, 0, 0]
        total_vi_quality = [0, 0, 0, 0, 0, 0]
        for i in range(num):
        # for i in range(1000, 1221):
        # for i in range(1000, 1040):
            index = i + 1
            # infrared_path = test_path + 'ir/'+'IR0' + str(index) + '.jpg'
            # visible_path = test_path + 'vis/'+'VIS0' + str(index) + '.jpg'
            infrared_path = '/media/ExtHDD/datasets/testfusion_images/IVT_test_TNO/ir/' + str(index) + '.png'
            visible_path = '/media/ExtHDD/datasets/testfusion_images/IVT_test_TNO/vis/' + str(index) + '.png'
            text_path = "/media/ExtHDD/datasets/testfusion_images/IVT_test_TNO/text/" + str(index) + "_5.txt"
            # infrared_path = '/media/ExtHDD/datasets/LLVIP3/ir/'+str(index) +'.png'
            # visible_path = '/media/ExtHDD/datasets/LLVIP3/vis/'+str(index) +'.png'
            # text_path = '/media/ExtHDD/datasets/LLVIP3/text/'+str(index) +'_1.txt'
            # infrared_path = '/media/ExtHDD/datasets/testfusion_images/IVT_test_LLVIP/ir/' + str(index) + '.png'
            # visible_path = '/media/ExtHDD/datasets/testfusion_images/IVT_test_LLVIP/vis/' + str(index) + '.png'
            # text_path = '/media/ExtHDD/datasets/testfusion_images/IVT_test_LLVIP/text/' + str(index) + '_3.txt'
            # infrared_path = '/media/ExtHDD/datasets/testfusion_images/IVT_test_RoadScene/ir/' + str(index) + '.png'
            # visible_path = '/media/ExtHDD/datasets/testfusion_images/IVT_test_RoadScene/vis/' + str(index) + '.png'
            # text_path = '/media/ExtHDD/datasets/testfusion_images/IVT_test_RoadScene/text/' + str(index) + '.txt'
            # load text content
            with open(text_path, 'r') as f:
               description = f.readline().strip()
               description = str(description)

            text = clip.tokenize([description]).to(device)
            # print(text.shape)
            xq,description_features = clip_model.encode_text(text)
            # description_features,xq = clip_model.encode_text(text)
            # print(xq.shape)

            # defog_path = test_path + 'results/' + 'VIS' + str(index) + '.png'
            defog_path=None
            # infrared_path = test_path + 'ir/' + 'IR' + str(index) + '.jpg'
            # visible_path = test_path + 'vis/' + 'VIS' + str(index) + '.jpg'
            # infrared_path = test_ir_path + 'roadscene' + '_' + str(index) + '.png'
            # visible_path = test_vis_path + 'roadscene' + '_' + str(index) + '.png'
            # infrared_path = test_ir_path + 'video' + '_' + str(index) + '.png'
            # visible_path = test_vis_path + 'video' + '_' +str(index) + '.png'
            print("第{}张图片融合成功".format(i + 1))
            run_demo(model, fusion_model, infrared_path, visible_path, output_path, index, description_features,visual)
        #     quality,h,w=run_demo(model,fusion_model ,infrared_path, visible_path,output_path, index,description_features )
        #     total_quality = [a + b for a, b in zip(quality, total_quality)]
        #     ir_quality = fix_feature_get.normal_quality(infrared_path,h,w)
        #     vi_quality = fix_feature_get.normal_quality(visible_path,h,w)
        #     total_ir_quality = [a + b for a, b in zip(ir_quality, total_ir_quality)]
        #     total_vi_quality = [a + b for a, b in zip(vi_quality, total_vi_quality)]
        # print("所有融合图像平均质量如下：")
        # fix_feature_get.total_qualiyt_show(total_quality, num)
        # print("红外图像平均质量如下：")
        # fix_feature_get.total_ir_show(total_ir_quality, num)
        # print("可见光图像平均质量如下：")
        # fix_feature_get.total_vi_show(total_vi_quality, num)


        end = time.time()
        print("consumption time of generating:%s " % (end - begin))
    print('Done......')


if __name__ == '__main__':
    main()