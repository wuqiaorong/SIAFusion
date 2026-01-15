# Training TextFusion network
import os
from densefuseNet import DenseFuse_net
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import time
from utils import gradient
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from net_patch12 import SwinFuse, FusionBlock_res
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


def main():
    # load pre-train models -------------- begin
    model_path_mask = '/media/ExtHDD/clipseg-master/weights/rd64-uni-refined.pth'
    mask_model = load_model_mask(model_path_mask)

    # clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(
        "/media/ExtHDD/pretrained_weights/ViT-B-32.pt"
        , device=device)

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

    patchPrePath = "/media/ExtHDD/datasets/LLVIP3/"
    PatchPaths = utils.generateTrainNumberIndex()
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
        fusion_model.train()
        # AE_model.train()
        print('Epoch %d.....' % e)
        patchesPaths, batches = utils.load_datasetPair(PatchPaths, batch_size)

        count = 0

        record_total_loss = 0.
        record_content_loss = 0.
        record_decomposition_loss = 0.
        record_cam_loss=0.


        batch = 0
        for i in range(batches):
            batch_inside = 0
            for textIndex in range(1, 1 + 5):
                image_paths = patchesPaths[batch_inside * batch_size:(batch_inside * batch_size + batch_size)]
                batch_inside += 1
                image_ir = utils.get_train_images_auto(patchPrePath + "ir/" + image_paths[0] + ".png",
                                                       height=args.height, width=args.width)
                image_vi = utils.get_train_images_auto(patchPrePath + "vis/" + image_paths[0] + ".png",
                                                       height=args.height, width=args.width)

                # # print(image_vi.shape)
                # binaryInterestedRegions,_ = generate_mask(mask_model,
                #                                                       patchPrePath + "ir/" + image_paths[0] + ".png",
                #                                                       patchPrePath + "vis/" + image_paths[0] + ".png",
                #                                                       patchPrePath + "text_words/" + image_paths[
                #                                                           0] + "_" + str(textIndex) + ".txt")
                binaryInterestedRegions = utils.get_train_images_auto_mask(
                    patchPrePath + "association/IVT_LLVIP_2000_imageIndex_" + image_paths[0]
                    + "_textIndex_" + str(textIndex)
                    + "/Final_Finetuned_BinaryInterestedMap.png",height=args.height, width=args.width)
                # print(binaryInterestedRegions.shape)
                binaryInterestedRegions = binaryInterestedRegions
                # print(torch.max(binaryInterestedRegions))

                ones = torch.ones_like(binaryInterestedRegions)
                binaryNonInterestedRegions = ones - binaryInterestedRegions

                # batch_size equals to one
                text_path = patchPrePath + "text/" + image_paths[0] + "_" + str(textIndex) + ".txt"
                # load defog text_feature
                # text_feature_path = patchPrePath + "text_feature/" + image_paths[0] + "_" + str(textIndex) + ".npy"
                # text_feature = np.load(text_feature_path)
                # text_feature=torch.from_numpy(text_feature).float()

                # load text content
                with open(text_path, 'r') as f:
                    description = f.readline().strip()
                    description = str(description)

                text = clip.tokenize([description]).to(device)
                _, description_features = clip_model.encode_text(text)

                h = image_ir.shape[2]
                w = image_ir.shape[3]

                fusion_model.zero_grad()

                if args.cuda:
                    image_ir = image_ir.cuda(args.device)
                    image_vi = image_vi.cuda(args.device)
                    binaryInterestedRegions = binaryInterestedRegions.cuda(args.device)
                    binaryNonInterestedRegions = binaryNonInterestedRegions.cuda(args.device)
                    description_feature = description_features.cuda(args.device).float()
                    # word_feature = word_feature.cuda(args.device)
                    # image_ir_target=image_ir_target.cuda(args.device)

                # non-interested regions get gradient-based measurement  ---begin
                with torch.no_grad():
                    dup_ir = torch.cat([image_ir, image_ir, image_ir], 1) / 255
                    dup_vi = torch.cat([image_vi, image_vi, image_vi], 1) / 255
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
                        tmpBinaryNonInterestedRegions = F.interpolate(tmpBinaryNonInterestedRegions, scale_factor=0.5)
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

                img1_en1, pos1 = AE_model.self_embed(image_vi, h, w)
                img1_en2 = AE_model.encoder(img1_en1)
                img1_en3 = fusion_model.encoder_vi(img1_en2)
                img1_texture, sim_vi,fea_cos_vi = fusion_model.encoder_text(img1_en3, description_feature)

                img2_en1, pos2 = AE_model.self_embed(image_ir, h, w)
                img2_en2 = AE_model.encoder(img2_en1)
                img2_en3 = fusion_model.encoder_ir(img2_en2)
                img2_texture, sim_ir,fea_cos_ir = fusion_model.encoder_text(img2_en3, description_feature)

                # interested regions get pixel-based measurement  ---begin
                # FeaturesIR = img2_en2
                # FeaturesVIS =img1_en2
                FeaturesIR = img2_en3
                FeaturesVIS =img1_en3
                # 上采样回归
                # proj = torch.nn.PixelShuffle(4)
                # FeaturesIR_1 = proj(FeaturesIR)
                # FeaturesVIS_1 = proj(FeaturesVIS)
                # interested regions get pixel-based measurement  ---begin
                denseFeaturesIR = densefuseModel.encoder(image_ir)[0]/255
                denseFeaturesVIS = densefuseModel.encoder(image_vi)[0]/255
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
                loss_cam = 1000*(0.5+l1_loss(sim_fusion_ir, sim_ir) + 0.5*l1_loss(sim_fusion_vi, sim_vi))
                # print(nonInterestedLoss)

                totalLoss = interestedLoss + nonInterestedLoss+loss_cam
                # Loss function definition ---end

                totalLoss.backward()
                optimizer_fusion_model.step()

                record_total_loss += totalLoss.item()
                record_content_loss += interestedLoss.item()
                record_decomposition_loss += nonInterestedLoss.item()
                record_cam_loss += loss_cam.item()
                step = step + 1
                # Append loss matrix
                if (batch + 1) % args.log_interval == 0:
                    mesg = "{}\tEpoch {}:\t[{}/{}]\t interestedLoss:{}\t noninterestedLoss:{}\t camLoss:{}\t TotalLoss:{}".format(
                        time.ctime(), e + 1, batch + 1, batches * 5, record_content_loss / args.log_interval,
                                      record_decomposition_loss / args.log_interval,record_cam_loss/args.log_interval
                        , record_total_loss / args.log_interval
                    )
                    tbar.set_description(mesg)
                    writer.add_scalar("total_loss", record_total_loss / args.log_interval, step)
                    writer.add_scalar("content_loss", record_content_loss / args.log_interval, step)
                    writer.add_scalar("decomposition_loss", record_decomposition_loss / args.log_interval, step)
                    writer.add_scalar("cam_loss", record_cam_loss / args.log_interval, step)

                    # Save loss model
                    if best_loss > record_total_loss:
                        best_loss = record_total_loss
                        # save model
                        fusion_model.eval()
                        fusion_model.cpu()
                        save_model_filename = "fusion_best_model_20250621_net_patch12" + ".model"
                        save_model_path = os.path.join(args.save_fusion_model, save_model_filename)
                        torch.save(fusion_model.state_dict(), save_model_path)

                        fusion_model.train()
                        if (args.cuda):
                            fusion_model.cuda(int(args.device))
                        tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)
                    record_total_loss = 0.
                    record_content_loss = 0.
                    record_decomposition_loss = 0.
                    record_cam_loss = 0.

                batch += 1
    fusion_model.eval()
    fusion_model.cpu()
    save_model_filename = "fusion_20250621_net_patch12_results_Epoch" + str(e + 1) + ".model"
    save_model_path = os.path.join(args.save_fusion_model, save_model_filename)
    torch.save(fusion_model.state_dict(), save_model_path)
    print("The final model saved at" + save_model_path)

    print("\nDone, trained model saved!")


if __name__ == "__main__":
    main()
