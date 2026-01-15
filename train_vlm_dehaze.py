# Training TextFusion network
import os
from densefuseNet import DenseFuse_net
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import time
from utils import gradient,load_dataset_aligned
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random

import torch
from torch.optim import Adam
from netvlm_defog4 import SwinFuse, FusionBlock_res
from torchvision.models import resnet50
import utils
from args_fusion import args
import pytorch_msssim
import torchvision.models as models
import torch.nn.functional as F
import clip
from mask_genrate_1 import load_model_mask, load_and_preprocess_image, generate_mask_from_prompts, combine_masks
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


def load_model(path,path2):
    in_c = 1
    nb_filter = 128
    in_chans = in_c
    out_chans = in_c
    denseFuseNet_model = DenseFuse_net()
    denseFuseNet_model.load_state_dict(torch.load(path2))
    AE_model = SwinFuse(in_chans=in_chans, out_chans=out_chans, img_size=420)
    AE_model.load_state_dict(torch.load(path))
    fusion_model = FusionBlock_res(in_chans=in_chans, out_chans=out_chans)
    if args.resume_fusion_model is not None:
        print('Resuming, initializing fusion net using weight from {}.'.format(args.resume_fusion_model))
        fusion_model.load_state_dict(torch.load(args.resume_fusion_model))

    para = sum([np.prod(list(p.size())) for p in AE_model.parameters()])
    print('Model {} : params: {:4f}M'.format(AE_model._get_name(), para / 1000 / 1000))
    para2 = sum([np.prod(list(p.size())) for p in denseFuseNet_model.parameters()])
    print('Model {} : params: {:4f}M'.format(denseFuseNet_model._get_name(), para2 / 1000 / 1000))

    AE_model.eval()
    fusion_model.eval()
    denseFuseNet_model.eval()

    return AE_model, fusion_model,denseFuseNet_model


def generate_mask(model, infrared_image_path, visible_image_path, prompts_file_path):
    infrared_img = load_and_preprocess_image(infrared_image_path)
    visible_img = load_and_preprocess_image(visible_image_path)

    infrared_mask, infrared_text_feature = generate_mask_from_prompts(model, infrared_img, prompts_file_path)
    visible_mask, visible_text_feature = generate_mask_from_prompts(model, visible_img, prompts_file_path)

    combined_mask = combine_masks(infrared_mask, visible_mask)
    combined_mask = torch.from_numpy(combined_mask).float().cuda()
    return combined_mask, visible_text_feature


def generate_mask_batch(model, infrared_image_paths, visible_image_paths, prompts_file_paths):
    batch_size = len(infrared_image_paths)
    combined_masks = []
    text_features = []

    for i in range(batch_size):
        infrared_img = load_and_preprocess_image(infrared_image_paths[i])
        visible_img = load_and_preprocess_image(visible_image_paths[i])

        infrared_mask, infrared_text_feature = generate_mask_from_prompts(model, infrared_img, prompts_file_paths[i])
        visible_mask, visible_text_feature = generate_mask_from_prompts(model, visible_img, prompts_file_paths[i])

        combined_mask = combine_masks(infrared_mask, visible_mask)
        combined_masks.append(combined_mask)
        text_features.append(visible_text_feature)
        # 添加保存掩码的代码
        # Image.fromarray((combined_mask * 255).astype(np.uint8)).save(f"mask_check/mask_{i}.png")

    # 将所有mask和text_feature堆叠成batch形式
    batch_combined_mask = np.stack(combined_masks, axis=0)
    batch_combined_mask = torch.from_numpy(batch_combined_mask).float().cuda()

    # 将text_feature堆叠成batch形式
    # batch_text_feature = torch.stack([torch.from_numpy(tf).float().cuda() for tf in text_features], dim=0)

    return batch_combined_mask

nan_printed = False
def main():
    # load pre-train models -------------- begin
    model_path_mask = '/media/ExtHDD/clipseg-master/weights/rd64-uni-refined.pth'
    mask_model = load_model_mask(model_path_mask)
    original_imgs_path_ir =utils.list_images(args.dataset_ir)
    original_imgs_path_vis = utils.list_images(args.dataset_vi)
    original_text_path= utils.list_images(args.dataset_text)
    original_text_mask_path=utils.list_images(args.dataset_text_mask)
    # original_text_path_ir = utils.list_images(args.dataset_vi_text)
    # clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(
        "/media/ExtHDD/pretrained_weights/ViT-B-32.pt"
        , device=device)
    clip_model.eval()

    # vgg
    vgg_model = models.vgg19(weights=None)
    vgg_model.load_state_dict(torch.load("/media/ExtHDD/pretrained_weights/vgg19-dcbb9e9d.pth"))
    vgg_model.eval()
    if (args.cuda):
        vgg_model = vgg_model.cuda(args.device)
    vggFeatures = []
    vggFeatures.append(vgg_model.features[:3])  # 64
    vggFeatures.append(vgg_model.features[:8])  # 32
    vggFeatures.append(vgg_model.features[:17])  # 16
    vggFeatures.append(vgg_model.features[:26])  # 8
    vggFeatures.append(vgg_model.features[:35])  # 4
    for i in range(0, 5):
        for parm in vggFeatures[i].parameters():
            parm.requires_grad = False

    # autoencoder
    model_path = args.model_path_gray
    model_path_dense = "models/DenseFuse.model"
    AE_model, fusion_model,densefuseModel = load_model(model_path,model_path_dense)
    if (args.cuda):
        AE_model = AE_model.cuda(args.device)
        fusion_model = fusion_model.cuda(args.device)
        densefuseModel = densefuseModel.cuda(args.device)
    # load pre-train models -------------- end

    # original_imgs_path_ir=utils.list_images(args.dataset_ir)
    batch_size = args.batch_size
    optimizer_fusion_model = Adam(fusion_model.parameters(), args.lr)

    mse_loss = torch.nn.MSELoss(reduction="mean")
    l1_loss = torch.nn.L1Loss(reduction="mean")
    # ssim_loss = pytorch_msssim.msssim
    # # if (args.cuda):
    # #     TextFusionNet_model.cuda(int(args.device))

    tbar = trange(args.epochs)
    print('Start training.....')
    step = 0  # 用于可视化误差
    writer = SummaryWriter("fusion_train")
    best_loss = 10000000.
    for e in tbar:
        # 替换原来的三行load_dataset调用为：
        image_set_ir, image_set_vi, text_mask_set,text_set, batches = load_dataset_aligned(
            original_imgs_path_ir, original_imgs_path_vis, original_text_mask_path,original_text_path, batch_size)

        fusion_model.train()
        # AE_model.train()
        print('Epoch %d.....' % e)


        count = 0

        record_total_loss = 0.
        record_content_loss = 0.
        record_decomposition_loss = 0.
        record_cam_loss=0.
        batch = 0
        for i in range(batches):
            image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
            image_paths_target = image_set_vi[batch * batch_size:(batch * batch_size + batch_size)]
            text_mask_path=text_mask_set[batch * batch_size:(batch * batch_size + batch_size)]
            text_path=text_set[batch * batch_size:(batch * batch_size + batch_size)]
            image_ir = utils.get_train_images_auto(image_paths_ir, height=args.height, width=args.width)
            # print(image_ir.shape)
            image_target = utils.get_train_images_auto(image_paths_target, height=args.height, width=args.width)
            image_paths_vi = [x.replace('infrared', 'visible_fog_shuttle') for x in image_paths_ir]
            image_vi = utils.get_train_images_auto(image_paths_vi, height=args.height, width=args.width)
            # print("Image paths IR:", image_paths_ir)
            # print("Image paths VI:", image_paths_vi)
            # print("Text paths:", text_path)
            # print(image_paths_ir)
            try:
                binaryInterestedRegions= generate_mask_batch(mask_model,image_paths_ir, image_paths_target,text_mask_path)
                # print("the pic ",i)
            except:
                print(text_path)

            binaryInterestedRegions = binaryInterestedRegions
            # print(binaryInterestedRegions.shape)
            # print(torch.max(binaryInterestedRegions))

            ones = torch.ones_like(binaryInterestedRegions)
            binaryNonInterestedRegions = ones - binaryInterestedRegions

            # load defog text_feature
            # text_feature_path = patchPrePath + "text_feature/" + image_paths[0] + "_" + str(textIndex) + ".npy"
            # text_feature = np.load(text_feature_path)
            # text_feature=torch.from_numpy(text_feature).float()

            # load text content batch
            descriptions = []
            for text_path_single in text_path:
                with open(text_path_single, 'r') as f:
                    description = f.readline().strip()
                    description = str(description)
                    descriptions.append(description)

            text = clip.tokenize(descriptions).to(device)
            _, description_features = clip_model.encode_text(text)

            # # load text content
            # with open(text_path, 'r') as f:
            #     description = f.readline().strip()
            #     description = str(description)
            #
            # text = clip.tokenize([description]).to(device)
            # _, description_features = clip_model.encode_text(text)

            h = image_ir.shape[2]
            w = image_ir.shape[3]

            fusion_model.zero_grad()

            if args.cuda:
                image_ir = image_ir.cuda(args.device)
                image_vi = image_vi.cuda(args.device)
                image_target=image_target.cuda(args.device)
                binaryInterestedRegions = binaryInterestedRegions.cuda(args.device)
                binaryNonInterestedRegions = binaryNonInterestedRegions.cuda(args.device)
                description_feature = description_features.cuda(args.device).float()
                # word_feature = word_feature.cuda(args.device)
                # image_ir_target=image_ir_target.cuda(args.device)

            # non-interested regions get gradient-based measurement  ---begin
            with torch.no_grad():
                dup_ir = torch.cat([image_ir, image_ir, image_ir], 1) / 255
                dup_vi = torch.cat([image_target, image_target, image_target], 1) / 255
                # dup_ir = torch.cat([image_ir, image_ir, image_ir], 1)
                # dup_vi = torch.cat([image_vi, image_vi, image_vi], 1)
                sum_g_ir = torch.zeros(1)
                sum_g_vi = torch.zeros(1)

                if args.cuda:
                    sum_g_ir = sum_g_ir.cuda(args.device)
                    sum_g_vi = sum_g_vi.cuda(args.device)
                depth_of_features = 5
                myscale = 1
                num = 5
                tmpBinaryNonInterestedRegions = binaryNonInterestedRegions / 255
                # tmpBinaryNonInterestedRegions = binaryNonInterestedRegions
                # print(tmpBinaryNonInterestedRegions.shape)
                for j in range(depth_of_features):
                    g_ir = gradient(vggFeatures[j](dup_ir)).pow(2)
                    g_vi = gradient(vggFeatures[j](dup_vi)).pow(2)

                    g_ir = g_ir.mean(dim=1, keepdim=True)
                    g_vi = g_vi.mean(dim=1, keepdim=True)
                    g_ir = g_ir * tmpBinaryNonInterestedRegions
                    g_vi = g_vi * tmpBinaryNonInterestedRegions

                    sum_non = torch.sum(tmpBinaryNonInterestedRegions)
                    if (sum_non.item() > 0):
                        sum_g_ir = sum_g_ir + torch.sum(g_ir) / sum_non
                        sum_g_vi = sum_g_vi + torch.sum(g_vi) / sum_non
                    # 获取当前尺寸
                    B, H, W = tmpBinaryNonInterestedRegions.shape
                    # 明确指定输出尺寸为原来的一半
                    tmpBinaryNonInterestedRegions = F.interpolate(tmpBinaryNonInterestedRegions.unsqueeze(1),
                                                                  size=(H // 2, W // 2),
                                                                  mode='nearest').squeeze(1)
                    # tmpBinaryNonInterestedRegions = F.interpolate(tmpBinaryNonInterestedRegions, scale_factor=0.5, mode='bilinear', align_corners=False)
                    # print("111111111111")
                    # print(tmpBinaryNonInterestedRegions.shape)
                sum_g_ir /= depth_of_features
                sum_g_vi /= depth_of_features

            sum_g_ir /= 4000
            sum_g_vi /= 4000

            weightNonInterestedIR = torch.exp(sum_g_ir) / (torch.exp(sum_g_ir) + torch.exp(sum_g_vi))
            # print(weightNonInterestedIR)
            weightNonInterestedVI = torch.exp(sum_g_vi) / (torch.exp(sum_g_ir) + torch.exp(sum_g_vi))
            # non-interested regions get gradient-based measurement  ---end

            # encoder net_patch2/3
            # en_vi_1 = AE_model.self_embed(image_vi,h,w)
            # en_ir_1 = AE_model.self_embed(image_ir,h,w)
            # img1, pos1 = fusion_model.pos_embed(en_vi_1, image_vi, description_feature, h, w)
            # img2, pos2 = fusion_model.pos_embed(en_ir_1, image_ir, description_feature, h, w)
            # img1_en1 = AE_model.encoder1(img1)
            # img1_en1 = fusion_model.encoder_text(img1_en1, pos1)
            # img1_en2 = AE_model.encoder1(img1_en1)
            # img1_en2 = fusion_model.encoder_text(img1_en2, pos1)
            #
            # img2_en1 = AE_model.encoder1(img2)
            # img2_en1 = fusion_model.encoder_text(img2_en1, pos2)
            # img2_en2 = AE_model.encoder1(img2_en1)
            # img2_en2 = fusion_model.encoder_text(img2_en2, pos2)
            # image_vi_pre = fusion_model.img_dazing(image_vi, image_ir)

            image_vi_pre, pos1 = AE_model.self_embed(image_vi, h, w)
            img2_en1, pos2 = AE_model.self_embed(image_ir, h, w)
            img1_en1 = fusion_model.img_dazing(image_vi_pre, img2_en1)
            img1_en2 = AE_model.encoder(img1_en1)

            img1_target_en1,pos_target=AE_model.self_embed(image_target, h, w)
            img1_target_en2 = AE_model.encoder(img1_target_en1)

            img1_texture, _,_ = fusion_model.encoder_text(img1_en2, description_feature)
            img1_target_texture, sim_vi, fea_cos_vi = fusion_model.encoder_text(img1_target_en2, description_feature)

            # img2_en1, pos2 = AE_model.self_embed(image_ir, h, w)
            img2_en2 = AE_model.encoder(img2_en1)
            img2_texture, sim_ir,fea_cos_ir = fusion_model.encoder_text(img2_en2, description_feature)

            # interested regions get pixel-based measurement  ---begin
            FeaturesIR = img2_en2
            FeaturesVIS =img1_en2
            # 上采样回归
            # proj = torch.nn.PixelShuffle(4)
            # FeaturesIR_1 = proj(FeaturesIR)
            # FeaturesVIS_1 = proj(FeaturesVIS)
            # interested regions get pixel-based measurement  ---begin
            denseFeaturesIR = densefuseModel.encoder(image_ir)[0]/255
            denseFeaturesVIS = densefuseModel.encoder(image_target)[0]/255
            # denseFeaturesIR = densefuseModel.encoder(image_ir)[0]
            # denseFeaturesVIS = densefuseModel.encoder(image_vi)[0]
            almIR = denseFeaturesIR.sum(dim=1, keepdim=True)
            almVIS = denseFeaturesVIS.sum(dim=1, keepdim=True)

            weightInterestedIR = torch.exp(almIR) / (torch.exp(almIR) + torch.exp(almVIS))

            # print(weightInterestedIR)
            weightInterestedVIS = torch.exp(almVIS) / (torch.exp(almIR) + torch.exp(almVIS))

            # interested regions get pixel-based measurement  ---end
            #net_patch2
            # fusedfeature = fusion_model(x_ir=FeaturesIR, x_vi=FeaturesVIS, text_features=description_feature
            #                             ,pos_vi=pos1,pos_ir=pos2)
            #net_patch3
            # fusedfeature = fusion_model(x_ir=FeaturesIR, x_vi=FeaturesVIS,pos_vi=pos1, pos_ir=pos2)
            fusedfeature = fusion_model(x_ir=FeaturesIR, x_vi=FeaturesVIS, pos_vi=img1_texture, pos_ir=img2_texture)
            sim_fusion_vi = fusion_model.loss_cam(fea_cos_vi, fusedfeature)
            sim_fusion_ir = fusion_model.loss_cam(fea_cos_ir, fusedfeature)


            fusedImage = AE_model.up_x4(fusedfeature)
            fusedImage = (fusedImage - torch.min(fusedImage)) / (
                        torch.max(fusedImage) - torch.min(fusedImage) + 1e-6)
            fusedImage = fusedImage * 255
            # Loss function definition ---begin
            interestedLoss = mse_loss(binaryInterestedRegions * weightInterestedIR * fusedImage,
                                      binaryInterestedRegions * weightInterestedIR * image_ir) + \
                             mse_loss(binaryInterestedRegions * weightInterestedVIS * fusedImage,
                                      binaryInterestedRegions * weightInterestedVIS * image_vi)
            # print(weightNonInterestedIR)
            # print(weightNonInterestedVI)
            # print(mse_loss(binaryNonInterestedRegions*fusedImage,binaryNonInterestedRegions*image_ir))
            # print(mse_loss(binaryNonInterestedRegions*fusedImage,binaryNonInterestedRegions*image_vi))
            nonInterestedLoss = weightNonInterestedIR * mse_loss(binaryNonInterestedRegions * fusedImage,
                                                                 binaryNonInterestedRegions * image_ir) + \
                                weightNonInterestedVI * mse_loss(binaryNonInterestedRegions * fusedImage,
                                                                 binaryNonInterestedRegions * image_vi)
            # loss_cam = 1000*(0.5+l1_loss(sim_fusion_ir, sim_ir) + 0.5*l1_loss(sim_fusion_vi, sim_vi))
            # print(nonInterestedLoss)

            # totalLoss = interestedLoss + nonInterestedLoss+loss_cam
            # 在计算totalLoss之后，添加检查NAN的代码
            totalLoss = interestedLoss + nonInterestedLoss

            # 检查是否有NAN值，如果有则打印图像路径
            # 替换你原来的NaN检查代码为以下代码
            global nan_printed
            # if (torch.isnan(totalLoss) or torch.isnan(interestedLoss) or torch.isnan(nonInterestedLoss) or torch.isnan(
            #         loss_cam)) and not nan_printed:
            if (torch.isnan(totalLoss) or torch.isnan(interestedLoss) or torch.isnan(
                    nonInterestedLoss) ) and not nan_printed:
                print("WARNING: NaN loss detected!")
                print("Image paths IR:", image_paths_ir)
                print("Image paths VI:", image_paths_vi)
                print("Text paths:", text_path)
                print("interestedLoss:", interestedLoss.item() if not torch.isnan(interestedLoss) else "NaN")
                print("nonInterestedLoss:", nonInterestedLoss.item() if not torch.isnan(nonInterestedLoss) else "NaN")
                # print("loss_cam:", loss_cam.item() if not torch.isnan(loss_cam) else "NaN")
                print("totalLoss:", totalLoss.item() if not torch.isnan(totalLoss) else "NaN")
                nan_printed = True

            # Loss function definition ---end

            totalLoss.backward()
            optimizer_fusion_model.step()
            record_total_loss += totalLoss.item()
            record_content_loss += interestedLoss.item()
            record_decomposition_loss += nonInterestedLoss.item()
            # record_cam_loss += loss_cam.item()
            step = step + 1
            # Append loss matrix
            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\t interestedLoss:{}\t noninterestedLoss:{}\t  TotalLoss:{}".format(
                    time.ctime(), e + 1, batch + 1, batches, record_content_loss / args.log_interval,
                                  record_decomposition_loss / args.log_interval, record_total_loss / args.log_interval
                )
                tbar.set_description(mesg)
                writer.add_scalar("total_loss", record_total_loss / args.log_interval, step)
                writer.add_scalar("content_loss", record_content_loss / args.log_interval, step)
                writer.add_scalar("decomposition_loss", record_decomposition_loss / args.log_interval, step)
                # writer.add_scalar("cam_loss", record_cam_loss / args.log_interval, step)

                # Save loss model
                if best_loss > record_total_loss:
                    best_loss = record_total_loss
                    # save model
                    fusion_model.eval()
                    fusion_model.cpu()
                    save_model_filename = "fusion_best_model_20251231_netvlm_defog4" + ".model"
                    save_model_path = os.path.join(args.save_fusion_model, save_model_filename)
                    torch.save(fusion_model.state_dict(), save_model_path)

                    fusion_model.train()
                    if (args.cuda):
                        fusion_model.cuda(int(args.device))
                    tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)
                record_total_loss = 0.
                record_content_loss = 0.
                record_decomposition_loss = 0.
                # record_cam_loss = 0.

            batch += 1
    fusion_model.eval()
    fusion_model.cpu()
    save_model_filename = "fusion_20251231_netvlm_defog4_results_Epoch" + str(e + 1) + ".model"
    save_model_path = os.path.join(args.save_fusion_model, save_model_filename)
    torch.save(fusion_model.state_dict(), save_model_path)
    print("The final model saved at" + save_model_path)

    print("\nDone, trained model saved!")


if __name__ == "__main__":
    main()
