import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np #魔改
from numpy import random
from torch.utils.data import Dataset, DataLoader

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

try:
    from ml.molecule_preprocessing import crop_xyxy_array, image_to_tensor
except ImportError:  # Direct execution from the ml directory.
    from molecule_preprocessing import crop_xyxy_array, image_to_tensor

#我自己的哲学代码
from tools.zeigen import class_name_getter
import gc


def _saved_class_id(predicted_index, predicted_class_name, class_indices=None):
    if class_indices and int(predicted_index) < len(class_indices):
        return int(class_indices[int(predicted_index)])
    try:
        return int(predicted_class_name)
    except (TypeError, ValueError):
        return int(predicted_index)


#-------------------------------------------------------------------
def detect(save_img=False):
    #原神，启动！
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / 'labels').mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    
    # 加载resnet模型和权重
    # 先加载权重文件以确定模型类型
    resnet_weights = torch.load(opt.resnet_dir+opt.resnet_name, map_location=device, weights_only=False)
    
    # 检查权重文件格式
    if 'model_state_dict' in resnet_weights:
        # .saving格式
        state_dict = resnet_weights["model_state_dict"]
        print("检测到.saving格式ResNet权重")
    else:
        # 尝试作为完整模型加载（.pt格式）
        print("尝试作为.pt格式ResNet模型加载")
        if hasattr(resnet_weights, 'state_dict'):
            # 完整模型对象
            state_dict = resnet_weights.state_dict()
        elif isinstance(resnet_weights, dict):
            # 可能是YOLO格式的ckpt，尝试获取model
            if 'model' in resnet_weights:
                state_dict = resnet_weights['model'].state_dict()
            elif 'ema' in resnet_weights:
                state_dict = resnet_weights['ema'].state_dict()
            else:
                # 无法识别格式，使用原权重
                state_dict = resnet_weights
        else:
            state_dict = resnet_weights
    
    # 检查权重文件中的键，判断是哪种ResNet模型
    if 'layer3.22.conv1.weight' in state_dict:
        # ResNet-101
        print("检测到ResNet-101权重，使用ResNet-101模型")
        resnet_model = models.resnet101(weights=None)
    elif 'layer3.5.conv1.weight' in state_dict:
        # 检查conv1的形状来判断是ResNet-50还是ResNet-34
        conv1_weight = state_dict.get('layer3.5.conv1.weight')
        if conv1_weight is not None:
            if conv1_weight.shape[2] == 1 and conv1_weight.shape[3] == 1:
                # Bottleneck (1x1 conv) - ResNet-50
                print("检测到ResNet-50权重，使用ResNet-50模型")
                resnet_model = models.resnet50(weights=None)
            else:
                # BasicBlock (3x3 conv) - ResNet-34
                print("检测到ResNet-34权重，使用ResNet-34模型")
                resnet_model = models.resnet34(weights=None)
        else:
            # 默认使用ResNet-50
            print("无法检测模型类型，默认使用ResNet-50模型")
            resnet_model = models.resnet50(weights=None)
    elif 'layer3.1.conv1.weight' in state_dict:
        # ResNet-18
        print("检测到ResNet-18权重，使用ResNet-18模型")
        resnet_model = models.resnet18(weights=None)
    else:
        # 尝试通过其他层的权重形状来判断
        if 'layer1.0.conv1.weight' in state_dict:
            conv1_weight = state_dict['layer1.0.conv1.weight']
            if conv1_weight.shape[2] == 3 and conv1_weight.shape[3] == 3:
                # BasicBlock (3x3 conv) - ResNet-18 or ResNet-34
                if 'layer3.5.conv1.weight' in state_dict:
                    print("检测到ResNet-34权重，使用ResNet-34模型")
                    resnet_model = models.resnet34(weights=None)
                else:
                    print("检测到ResNet-18权重，使用ResNet-18模型")
                    resnet_model = models.resnet18(weights=None)
            else:
                # Bottleneck (1x1 conv) - ResNet-50 or ResNet-101
                if 'layer3.22.conv1.weight' in state_dict:
                    print("检测到ResNet-101权重，使用ResNet-101模型")
                    resnet_model = models.resnet101(weights=None)
                else:
                    print("检测到ResNet-50权重，使用ResNet-50模型")
                    resnet_model = models.resnet50(weights=None)
        else:
            # 默认使用ResNet-50
            print("无法检测模型类型，默认使用ResNet-50模型")
            resnet_model = models.resnet50(weights=None)
    
    # 从权重中获取类别数量（用于自适应）
    weight_class_num = state_dict['fc.weight'].shape[0] if 'fc.weight' in state_dict else 10
    
    # 尝试加载类别信息 YAML 文件
    import os
    class_type = [str(name) for name in resnet_weights.get('class_names', [])]
    class_indices = [int(index) for index in resnet_weights.get('class_indices', [])]
    class_num = 0
    use_yaml = opt.classes_yaml and os.path.exists(opt.classes_yaml)
    
    if use_yaml:
        try:
            import yaml
            with open(opt.classes_yaml, 'r', encoding='utf-8') as f:
                classes_info = yaml.safe_load(f)
            if 'names' in classes_info:
                class_type = [str(name) for name in classes_info['names']]
                class_num = len(class_type)
                indices = classes_info.get('indices')
                if isinstance(indices, list) and len(indices) == class_num:
                    class_indices = [int(index) for index in indices]
                print(f"从 YAML 文件加载类别信息成功")
                print(f"类别数: {class_num}")
                print(f"类别名称: {class_type}")
        except Exception as e:
            print(f"加载类别信息文件失败: {e}")
            use_yaml = False
    
    if not use_yaml and class_type:
        class_num = len(class_type)
        print(f"从 ResNet 检查点加载类别信息: {class_type}")
    elif not use_yaml:
        # 没有YAML文件，从权重中自动识别类别数
        class_num = weight_class_num
        class_type = [str(i) for i in range(class_num)]
        print(f"从 ResNet 模型自动识别类别数: {class_num}")
        print(f"类别名称: {class_type}")
    
    if class_num != weight_class_num:
        raise ValueError(f"类别数量不匹配：ResNet 输出 {weight_class_num} 类，类别信息包含 {class_num} 类")
    if class_indices and len(class_indices) != class_num:
        raise ValueError("ResNet 检查点中的 class_indices 与类别数量不一致")

    # 设置模型输出层并加载权重
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = torch.nn.Linear(num_ftrs, class_num)
    resnet_model.load_state_dict(state_dict)
    resnet_model.to(device)
    resnet_model.eval()
    
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors, set the former 6 as fixed
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_type]
    Couleurs=[[0,0,255],[255,0,0],[0,215,255],[0,255,0],[128,0,128],[128,128,128]]
    for Farbe in range(len(colors)):
        if Farbe<=len(Couleurs)-1:
            colors[Farbe]=Couleurs[Farbe]
        else:
            pass
    
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    
    # 获取数据集大小
    dataset_size = len(dataset)
    print(f"总共 {dataset_size} 张图片")
    
    # 遍历所有图片
    for img_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        print(f"\nimage {img_idx+1}/{dataset_size} {path}")
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            # Convert OpenCV BGR input once; plotting, PIL saving and classification all use RGB.
            im0=cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            classification_image = im0.copy()
            Chiffres=np.zeros(class_num) #存储各个图片的种类
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                #魔改
                index=0
                total_boxes = len(det)
                print(f"检测到 {total_boxes} 个框")
                
                for *xyxy, conf, cls in reversed(det):
                    index+=1
                    
                    # 计算框级进度
                    box_progress = (img_idx + (index / total_boxes)) / dataset_size * 100
                    
                    if save_img or view_img or save_txt:
                        crop = crop_xyxy_array(classification_image, xyxy)
                        crop_tensor = image_to_tensor(crop).unsqueeze(0).to(device)
                        with torch.no_grad():
                            probabilities = torch.softmax(resnet_model(crop_tensor), dim=1)
                        class_confidence, predicted = probabilities.max(dim=1)
                        predicted_index = int(predicted.item())
                        predicted_class_name = str(class_type[predicted_index])
                        class_confidence = float(class_confidence.item())
                        detection_confidence = float(conf.item())
                        is_confident = class_confidence >= opt.class_conf_thres
                        display_name = predicted_class_name if is_confident else "UNK"
                        label = f"{display_name} D{detection_confidence:.2f} C{class_confidence:.2f}"
                        Chiffres[predicted_index] += 1

                        print(
                            f"{index}/{total_boxes} type:{display_name} "
                            f"d={detection_confidence:.3f} c={class_confidence:.3f}"
                        )
                        image_extent = max(im0.shape[:2])
                        plot_one_box(
                            xyxy,
                            im0,
                            label=label,
                            color=colors[predicted_index],
                            line_thickness=max(1, round(image_extent / 512)),
                            label_scale=max(0.35, min(0.55, image_extent / 1600)),
                        )

                        if save_txt:
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            saved_class_id = _saved_class_id(predicted_index, predicted_class_name, class_indices)
                            line = (
                                saved_class_id,
                                *xywh,
                                detection_confidence,
                                class_confidence,
                            ) if opt.save_conf else (saved_class_id, *xywh)
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # Print time (inference + NMS), print class
            t3 = time_synchronized()
            print(Chiffres,str(np.sum(Chiffres)))
            print(f'{s}Done. ({t3 - t1:.3f}s)')
            
            # 保存检测结果到txt文件（使用ResNet分类后的类别）
            if save_txt:
                with open(txt_path + '.txt', 'a') as f:
                    f.write(str(Chiffres))
                    f.write('\n')
                    f.write(str(np.sum(Chiffres)))
                    f.write('\n')
                    f.close()

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    # 使用PIL保存图片，更好地支持中文路径
                    try:
                        from PIL import Image
                        # 转换BGR到RGB
                        # if im0.shape[2] == 3:
                            # im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                        # 使用PIL保存
                        pil_img = Image.fromarray(im0)
                        pil_img.save(save_path)
                        print(f"图片保存成功: {save_path}")
                    except Exception as e:
                        print(f"PIL保存失败，尝试使用cv2保存: {e}")
                        # 尝试使用cv2保存作为备选
                        if cv2.imwrite(save_path, im0):
                            print(f"cv2保存成功: {save_path}")
                        else:
                            print(f"保存失败: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

#E:\wzx\AI and machine learning\STM\Yolo\yolov5_5p0-20240409T065957Z-001\yolov5_5p0\dataset24_fp\images\val
if __name__ == '__main__':#20240416_purged1/JPEGImages_train uint8
    parser = argparse.ArgumentParser()#/weights
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/demo/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='dataset/low_reso', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold') #0.25
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--class-conf-thres', type=float, default=0.5, help='classification confidence threshold')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='test', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--mag', default=1,type=float, help='magnification ratio of STM images')
    parser.add_argument('--resnet_dir', default="runs/resnet101/",type=str, help='saved resnet-101 path')
    parser.add_argument('--resnet_name', default="demo.saving",type=str, help='saved resnet-101 name')
    parser.add_argument('--classes_yaml', default="",type=str, help='path to classes.yaml file for class names')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
