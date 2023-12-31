import os
import rospy
import copy
import torch
import cv_bridge
import sensor_msgs
import gc
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

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

    # assert the size of color image and depth image are both 1280*720, otherwise through an error
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
    o3d.visualization.draw_geometries([pcd])

    # mask the target object in depth image
    width, height = depth_array.shape[1], depth_array.shape[0]
    print("image width and height:", width, height)
    mask = mask.squeeze(0).numpy()
    for y in range(height):
          for x in range(width):
                if mask[y, x] == 0:
                 depth_array[y, x] = 0
    # create target point cloud from the masked depth image
    target_pcd = o3d.geometry.PointCloud.create_from_depth_image(rgbd_image.depth, camera_intrinsic)
    #target_pcd = target_pcd.voxel_down_sample(0.05)
    # point cloud clustering
    #with o3d.utility.VerbosityContextManager(
    #        o3d.utility.VerbosityLevel.Debug) as cm:
    #    labels = np.array(
    #        target_pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

    #max_label = labels.max()
    #print(f"point cloud has {max_label + 1} clusters")
    #colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    #colors[labels < 0] = 0
    #target_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # visualize the target point cloud
    #o3d.visualization.draw_geometries([target_pcd])

    return target_pcd, pcd


def normalize_vector(v):
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def rotation_matrix_from_normal(normal):
    """Generate a 3x3 rotation matrix from a normal vector."""
    normal = normalize_vector(normal)

    # Check if the normal vector is already aligned with the z-axis
    if np.allclose(normal, [0, 0, 1]):
        return np.identity(3)

    # Calculate the rotation axis using cross product
    rotation_axis = np.cross([0, 0, 1], normal)
    rotation_axis = normalize_vector(rotation_axis)

    # Calculate the rotation angle using dot product
    rotation_angle = np.arccos(np.dot([0, 0, 1], normal))

    # Generate the rotation matrix using the axis-angle representation
    c = np.cos(rotation_angle)
    s = np.sin(rotation_angle)
    t = 1 - c

    rotation_matrix = np.array([
        [t * rotation_axis[0] ** 2 + c, t * rotation_axis[0] * rotation_axis[1] - s * rotation_axis[2],
         t * rotation_axis[0] * rotation_axis[2] + s * rotation_axis[1]],
        [t * rotation_axis[0] * rotation_axis[1] + s * rotation_axis[2], t * rotation_axis[1] ** 2 + c,
         t * rotation_axis[1] * rotation_axis[2] - s * rotation_axis[0]],
        [t * rotation_axis[0] * rotation_axis[2] - s * rotation_axis[1],
         t * rotation_axis[1] * rotation_axis[2] + s * rotation_axis[0], t * rotation_axis[2] ** 2 + c]
    ])

    return rotation_matrix

def get_target_center(depth_image, rgb_image, masks):
    '''
    Get the center coordinate of the target point cloud
    :param masks: detected masks of RGB image
    :param boxes: detected 2D bounding boxes of RGB image
    :param depth_image: input depth image
    :return: calculated target center of the query object
    '''

    # create target point cloud from depth image with a mask
    target_pcd, pcd = create_target_pointcloud(depth_image, rgb_image, masks[0])

    # calculate the center of the target point cloud
    target_pcd_center = target_pcd.get_center()

    print("target_pcd_center:", target_pcd_center)
    center_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    center_marker.translate(target_pcd_center)
    center_marker.paint_uniform_color([1, 0, 0]) #red

    # Visualize the point cloud and the center marker
    o3d.io.write_point_cloud("/home/eggsy/Soft-Man/experiment/Round1/whole_pcd.ply", pcd, write_ascii=False, compressed=False, print_progress=False)
    o3d.io.write_point_cloud("/home/eggsy/Soft-Man/experiment/Round1/target_pcd.ply", target_pcd, write_ascii=False, compressed=False, print_progress=False)
    o3d.visualization.draw_geometries([pcd, center_marker], window_name="Point Cloud with Center")

    # Find PCD center's nearest point on PCD
    target_pcd_tree = o3d.geometry.KDTreeFlann(target_pcd)
    print("Calculating the nearest point on mesh......")
    center_mesh = target_pcd_tree.search_knn_vector_3d(target_pcd_center, 1)
    num_points_found, indices, distances = center_mesh
    mesh_marker_coords = np.asarray(target_pcd.points)[indices[0]]
    mesh_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    mesh_marker.translate(mesh_marker_coords)
    mesh_marker.paint_uniform_color([1, 0, 0]) #blue
    print("grasp point on mesh is:", mesh_marker_coords)

    # Find grasp point's normal vector
    target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)) #unit:cm
    o3d.geometry.PointCloud.orient_normals_towards_camera_location(target_pcd, camera_location=np.array([0, 0, 0]))
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    FOR2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=mesh_marker_coords)
    NOR = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.015, cylinder_height=0.07, cone_height=0.04, resolution=20, cylinder_split=4, cone_split=1)
    NOR.translate(mesh_marker_coords)
    normal_vec = -1 * target_pcd.normals[indices[0]]
    R = rotation_matrix_from_normal(normal_vec)
    NOR.rotate(R, center=mesh_marker_coords)
    NOR.paint_uniform_color([0, 0, 1])
    FOR2.rotate(R, center=mesh_marker_coords)

    # Given the Transformation Matrix
    R = np.array([[R[0][0], R[0][1], R[0][2], 0],
                     [R[1][0], R[1][1], R[1][2], 0],
                     [R[2][0], R[2][1], R[2][2], 0],
                     [0, 0, 0, 1]])
    Translation_matrix = np.array([
        [1, 0, 0, mesh_marker_coords[0]],
        [0, 1, 0, mesh_marker_coords[1]],
        [0, 0, 1, mesh_marker_coords[2]],
        [0, 0, 0, 1]
    ])
    Transformation_matrix = np.dot(Translation_matrix, R)
    print("Transformation Matrix from Camera Frame:", Transformation_matrix)
    #print("grasp pose:", mesh_marker_coords, R, "frame: camera")
    o3d.visualization.draw_geometries([NOR, FOR1, pcd, FOR2, mesh_marker], window_name="Point Cloud with Point on Mesh", point_show_normal=True)

    return target_pcd_center, mesh_marker_coords

def get_rgbd_image_and_camera_intrinsic():
    rospy.init_node("get_image")
    bridge = cv_bridge.CvBridge()

    #get rgb image
    print("Getting RGB image......")
    rgb_image = rospy.wait_for_message("/camera/color/image_raw", sensor_msgs.msg.Image)
    rgb_image = bridge.imgmsg_to_cv2(rgb_image,"rgb8")
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    #cv2.imwrite("/home/eggsy/Soft-Man/dataset/assets/rgb_image/rgb_img.png", rgb_image)
    cv2.imwrite("/home/eggsy/Soft-Man/experiment/Round1/rgb_img.png", rgb_image)
    rospy.loginfo("RGB image saved.")

    #get depth image
    depth_image = rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", sensor_msgs.msg.Image)
    print("Getting Depth image......")
    depth_image = bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough")
    #cv2.imwrite("/home/eggsy/Soft-Man/dataset/assets/depth_image/depth_img.png", depth_image)
    cv2.imwrite("/home/eggsy/Soft-Man/experiment/Round1/depth_img.png", depth_image)
    rospy.loginfo("Depth image saved.")
def reconstruct():
    pass


if __name__ == '__main__':
    # input text prompt and image
    #get_rgbd_image_and_camera_intrinsic()
    rgb_image = "/home/eggsy/Soft-Man/experiment/Round1/rgb_img.png"
    depth_image = "/home/eggsy/Soft-Man/experiment/Round1/depth_img.png"
    # text = input("Please input your query: ")
    text = "white bag"

    # run object detection
    masks, boxes = detect(text, rgb_image)

    # run object center targeting
    target_pcd_center = get_target_center(depth_image, rgb_image, masks)


