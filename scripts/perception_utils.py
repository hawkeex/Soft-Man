import os
import copy
import torch
import gc
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
import csv

# OwlViT Detection
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# segment anything
from segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_owlvit(checkpoint_path="owlvit-large-patch14", device='cuda'):
    """
    Return: model, processor (for text inputs)
    """
    processor = OwlViTProcessor.from_pretrained(f"google/{checkpoint_path}")
    model = OwlViTForObjectDetection.from_pretrained(f"google/{checkpoint_path}")
    model.to(device)
    model.eval()

    return model, processor


def detect(text_prompt, rgb_image):
    '''
    Detect the target object in the RGB image according to the text prompt
    :param text_prompt: detected masks of RGB image
    :param rgb_image: detected 2D bounding boxes of RGB image
    :return: detected bounding boxes and semantic masks of the target object
    '''

    # load image & texts
    image_pth = rgb_image
    image = Image.open(rgb_image)
    texts = [text_prompt.split(",")]

    # make dir
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # load OWL-ViT model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    owlvit_model = "owlvit-base-patch32"
    model, processor = load_owlvit(checkpoint_path=owlvit_model, device=device)
    box_threshold = 0.0
    get_topk =True

    # run object detection model
    with torch.no_grad():
        inputs = processor(text=texts, images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process_object_detection(outputs=outputs, threshold=box_threshold,
                                                      target_sizes=target_sizes.to(device))

    scores = torch.sigmoid(outputs.logits)
    topk_scores, topk_idxs = torch.topk(scores, k=1, dim=1)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    if get_topk:
        topk_idxs = topk_idxs.squeeze(1).tolist()
        topk_boxes = results[i]['boxes'][topk_idxs]
        topk_scores = topk_scores.view(len(text), -1)
        topk_labels = results[i]["labels"][topk_idxs]
        boxes, scores, labels = topk_boxes, topk_scores, topk_labels
    else:
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    # Print detected objects and rescaled box coordinates
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

    boxes = boxes.cpu().detach().numpy()
    normalized_boxes = copy.deepcopy(boxes)

    # visualize pred
    size = image.size
    pred_dict = {
        "boxes": normalized_boxes,
        "size": [size[1], size[0]],  # H, W
        "labels": [text[idx] for idx in labels]
    }

    # release the OWL-ViT
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # run segment anything (SAM)
    predictor = SamPredictor(build_sam(checkpoint="model/sam_vit_h_4b8939.pth"))
    image = cv2.imread(image_pth)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    H, W = size[1], size[0]

    for i in range(boxes.shape[0]):
        boxes[i] = torch.Tensor(boxes[i])

    boxes = torch.tensor(boxes, device=predictor.device)

    transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2])

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    plt.figure(figsize=(12.8, 7.2))
    plt.imshow(image)
    for mask in masks:
        print("Mask shape before processing:", mask.numpy().shape)
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box in boxes:
        show_box(box.numpy(), plt.gca())
    plt.axis('off')
    plt.savefig(f"./{output_dir}/owlvit_segment_anything_output.png")

    # grounded results
    image_pil = Image.open(image_pth)
    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    image_with_box.save(os.path.join(f"./{output_dir}/owlvit_box.png"))

    return masks, boxes


def create_target_pointcloud(depth_image, rgb_image, mask):
    '''
    Create the point cloud of target object given the RGB-D image and the mask.
    :param depth_image: input depth image
    :param rgb_image: input RGB image
    :param mask: detected masks of the target object
    :return: created target point cloud
    '''

    # create rgbd image
    color_raw = o3d.io.read_image(rgb_image)
    depth_raw = o3d.io.read_image(depth_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

    color_array = np.asarray(rgbd_image.color)
    depth_array = np.asarray(rgbd_image.depth)

    # assert the size of color image and depth image are both 848*480, otherwise through an error
    assert color_array.shape[0] == depth_array.shape[0] and color_array.shape[1] == depth_array.shape[1], "The size of color image and depth image are not the same."

    # show rgbd image and point cloud
    plt.subplot(1, 2, 1)
    plt.title('RGB image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()

    # create point cloud from rgbd image
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(1280, 720, 635.861023, 635.861023, 641.331360, 349.101257)
    # camera_intrinsic = CameraIntrinsic(848, 480, 421.2579, 421.2579, 424.8820, 232.7795)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

    # visualize the point cloud
    #o3d.visualization.draw_geometries([pcd])

    # mask the target object in depth image
    width, height = depth_array.shape[1], depth_array.shape[0]
    print(width, height)
    mask = mask.squeeze(0).numpy()
    for y in range(height):
          for x in range(width):
                if mask[y, x] == 0:
                 depth_array[y, x] = 0
    print('1')
    # create target point cloud from the masked depth image
    target_pcd = o3d.geometry.PointCloud.create_from_depth_image(rgbd_image.depth, camera_intrinsic)

    # point cloud clustering
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            target_pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    target_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # visualize the target point cloud
    #o3d.visualization.draw_geometries([target_pcd])

    return target_pcd


def get_target_center(depth_image, rgb_image, masks):
    '''
    Get the center coordinate of the target point cloud
    :param masks: detected masks of RGB image
    :param boxes: detected 2D bounding boxes of RGB image
    :param depth_image: input depth image
    :return: calculated target center of the query object
    '''

    # create target point cloud from depth image with a mask
    target_pcd = create_target_pointcloud(depth_image, rgb_image, masks[0])

    # calculate the center of the target point cloud
    target_pcd_center = target_pcd.get_center()

    print("target_pcd_center:", target_pcd_center)

    return target_pcd_center


def reconstruct():
    pass


if __name__ == '__main__':

    # input text prompt and image
    rgb_image = "/home/eggsy/Soft-Man/dataset/rgb/out/001_Color.png"
    depth_image = "/home/eggsy/Soft-Man/dataset/depth/out/001_Depth.png"
    # text = input("Please input your query: ")
    text = "white bag"

    # run object detection
    masks, boxes = detect(text, rgb_image)

    # run object targeting
    target_pcd_center = get_target_center(depth_image, rgb_image, masks)



