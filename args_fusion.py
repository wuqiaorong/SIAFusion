class args():
    # training args
    epochs = 12#默认值是50,这里作5轮迁移学习
    batch_size = 1
    in_chans = 1
    out_chans = 1
    embed_dim = 512
    window_size = 7
    device = 0
    dataset = "/media/ExtHDD/datasets/coco2017/train2017"

    dataset_ir = "/media/ExtHDD/fusion_dataset/LLVIP2/infrared"
    dataset_vi = "/media/ExtHDD/fusion_dataset/LLVIP2/visible"
    dataset_text = "/media/ExtHDD/SwinFuse_new/tools/LLVIP/visible"
    dataset_text_mask="/media/ExtHDD/fusion_dataset/LLVIP2/LLVIP2_word/visible"
    # dataset_vi_text = "/root/autodl-tmp/data3/LLVIP2/visible"

    dataset_fog = "/root/autodl-tmp/data3/LLVIP2/visible_fog_shuttle"
    # dataset_ir = "/root/autodl-tmp/data2/KAIST/infrared"
    # dataset_vi = "/root/autodl-tmp/data2/KAIST/visible"

    save_fusion_model = "/media/ExtHDD/SwinFuse_new/models/fusion_vlm/"
    save_net_model = "/root/autodl-tmp/swim_new/models/net/"
    save_fog_model="/root/autodl-tmp/swim_new/models/defog/"
    save_model_dir = "models/restormer_base"  # "path to folder where trained model will be saved."
    trainNumber = 2000
    height = 416
    width =416
    image_size = [416,416] # "size of training images, default is 224 X 224"
    cuda = 1  # "set it to 1 for running on GPU, 0 for CPU"
    seed = 42
    ssim_weight = [1, 10, 100, 1000, 10000]
    grad_weight = 10
    ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']

    lr = 1e-4  # "learning rate"
    lr_light = 1e-5  # "learning rate"
    log_interval = 5  # "number of images after which the training loss is logged"
    log_iter = 1
    # resume = "/root/autodl-tmp/swim_new/models/1e3/Final_epoch_7_Wed_Dec_13_08_04_47_2023_1e3.model"
    resume=None
    resume_auto_en = None
    resume_auto_de = None
    resume_auto_fn = None
    resume_fusion_model =None
    resume_defog=None

    #vlm_dehaze
    model_fusion="/media/ExtHDD/SwinFuse_new/models/fusion_vlm/fusion_20251021_netvlm_defog3_results_Epoch12.model"

    # net_patch
    model_path_gray="/media/ExtHDD/SwinFuse_new/models/restormer_base/best_net_patch5.model"
    model_path_dense="./test_othe_model/models/Final_epoch_4_Thu_Nov__2_10_52_05_2023_1e2.model"
    model_default="/root/autodl-tmp/swim_new/test_othe_model/models/nestfuse_1e2.model"
    fog_path="/root/autodl-tmp/swim_new/models/hazing/Myfoggy3_epoch_10Tue_Mar_26_17_50_36_2024.model"
    salient_path="/root/autodl-tmp/swim_new/models/hazing/Mysalient2_epoch_10Mon_Apr_22_15_22_22_2024.model"