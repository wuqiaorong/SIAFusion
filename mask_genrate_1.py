import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from clipseg import CLIPDensePredT
import ast


def load_model_mask(model_path):
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
    model.eval()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')), strict=False)
    model = model.cuda()
    return model


def load_and_preprocess_image(image_path):
    input_image = Image.open(image_path)
    input_image = input_image.convert('RGB')  # 将灰度图像转换为三通道图像
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((420, 420), antialias=True),
    ])
    img = transform(input_image).unsqueeze(0).cuda()
    return img


def generate_aggregated_mask(model, img, prompts, threshold=0.5, min_threshold=0.001):
    with torch.no_grad():
        preds, _, text_feature, _ = model(img.repeat(len(prompts), 1, 1, 1), prompts, return_features=True)
        H, W = preds.shape[2], preds.shape[3]
        aggregated_mask = torch.zeros((1, H, W)).cuda()

        for i in range(preds.shape[0]):
            a = torch.sigmoid(preds[i][0])
            if torch.max(a) < min_threshold:
                continue
            a = (a - torch.min(a)) / (torch.max(a) - torch.min(a))
            a_binary = (a > threshold).float()
            aggregated_mask += a_binary

    aggregated_mask_np = aggregated_mask.cpu().numpy()
    aggregated_mask_np = (aggregated_mask_np * 255).astype(np.uint8)
    return aggregated_mask_np, text_feature


def save_image(mask, output_path):
    # 将 numpy 数组转换为 PIL 图像
    mask = Image.fromarray(mask.astype(np.uint8))
    # image = Image.fromarray(mask)
    # print(image)
    # print(mask.shape)
    # if image.mode == 'F':
    #     image = image.convert('L')
    mask.save(output_path)


def read_prompts_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        prompts = ast.literal_eval(content)
        salient_prompts = prompts[0]
        texture_prompts = prompts[1]
    return salient_prompts, texture_prompts


def generate_mask_from_prompts(model, img, file_path):
    salient_prompts, texture_prompts = read_prompts_from_file(file_path)

    if salient_prompts:
        mask, text_feature = generate_aggregated_mask(model, img, salient_prompts)
    elif texture_prompts:
        mask, text_feature = generate_aggregated_mask(model, img, texture_prompts)
    else:
        H, W = img.shape[2], img.shape[3]
        mask = np.ones((1, H, W), dtype=np.uint8) * 255
        text_feature = torch.zeros((1, 512))  # 创建一个全0的张量，形状为 [1, 512]

    return mask, text_feature


def combine_masks(mask1, mask2):
    combined_mask = np.maximum(mask1, mask2)
    ned_mask=Image.fromarray(combined_mask[0])
    ned_mask=ned_mask.resize((416,416),Image.Resampling.LANCZOS)
    ned_mask=np.array(ned_mask)/255.0
    return ned_mask


if __name__ == "__main__":
    model_path = '/media/ExtHDD/clipseg-master/weights/rd64-uni-refined.pth'
    infrared_image_path = "/media/ExtHDD/datasets/LLVIP3/ir/3.png"
    visible_image_path = "/media/ExtHDD/datasets/LLVIP3/vis/3.png"
    output_path = 'combined_output_image.png'
    prompts_file_path = "/media/ExtHDD/datasets/LLVIP3/text_words/3_5.txt"

    model = load_model_mask(model_path)

    infrared_img = load_and_preprocess_image(infrared_image_path)
    visible_img = load_and_preprocess_image(visible_image_path)
    # infrared_img=infrared_img.cuda()
    # visible_img=visible_img.cuda()

    infrared_mask, infrared_text_feature = generate_mask_from_prompts(model, infrared_img, prompts_file_path)
    visible_mask, visible_text_feature = generate_mask_from_prompts(model, visible_img, prompts_file_path)

    combined_mask = combine_masks(infrared_mask, visible_mask)
    # combined_mask=combined_mask.cpu()

    save_image(combined_mask*255, output_path)
    print(infrared_text_feature.shape)
    print(visible_text_feature.shape)
