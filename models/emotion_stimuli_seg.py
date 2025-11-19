import torch
from PIL import Image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import groundingdino.datasets.transforms as TG
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict

from segment_anything import build_sam, SamPredictor
from segment_anything.utils.amg import remove_small_regions

import numpy as np

groundingdino_config_file = "./GroundingDINO_SwinT_OGC.py"
groundingdino_checkpoint = "./models/groundingdino_swint_ogc.pth"
sam_checkpoint = "./models/sam_vit_h_4b8939.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_groundingdino_model(model_config_path, model_checkpoint_path):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

grounding_model = load_groundingdino_model(groundingdino_config_file, groundingdino_checkpoint).to(device)
sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device=device))

def prompt2mask(original_image, caption, box_threshold=0.25, text_threshold=0.25, num_boxes=2):
    # turn to pil if original_image is numpy array
    if isinstance(original_image, np.ndarray):
        original_image = Image.fromarray(original_image)
    else:
        original_image = original_image.convert('RGB')

    def image_transform_grounding(init_image):

        transform = TG.Compose([
            TG.RandomResize([800], max_size=1333),
            TG.ToTensor(),
            TG.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image, _ = transform(init_image, None)  # 3, h, w
        return init_image, image

    image_np = np.array(original_image, dtype=np.uint8)
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    _, image_tensor = image_transform_grounding(original_image)
    boxes, logits, phrases = predict(grounding_model,
                                     image_tensor, caption, box_threshold, text_threshold, device='cpu')

    # exit(0)
    # from PIL import Image, ImageDraw, ImageFont
    H, W = original_image.size[1], original_image.size[0]
    boxes = boxes * torch.Tensor([W, H, W, H])
    boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
    boxes[:, 2:] = boxes[:, 2:] + boxes[:, :2]
    # draw = ImageDraw.Draw(original_image)
    # for box in boxes:
    #     # from 0..1 to 0..W, 0..H
    #     # box = box * torch.Tensor([W, H, W, H])
    #     # # from xywh to xyxy
    #     # box[:2] -= box[2:] / 2
    #     # box[2:] += box[:2]
    #     # random color
    #     color = tuple(np.random.randint(0, 255, size=3).tolist())
    #     # draw
    #     x0, y0, x1, y1 = box
    #     x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    #
    #     draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
    # original_image.save('debug.jpg')
    # exit(0)

    final_m = torch.zeros((image_np.shape[0], image_np.shape[1]))

    if boxes.size(0) > 0:
        sam_predictor.set_image(image_np)

        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes, image_np.shape[:2])
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(device),
            multimask_output=False,
        )

        # remove small disconnected regions and holes
        fine_masks = []
        for mask in masks.to('cpu').numpy():  # masks: [num_masks, 1, h, w]
            fine_masks.append(remove_small_regions(mask[0], 400, mode="holes")[0])
        masks = np.stack(fine_masks, axis=0)[:, np.newaxis]
        masks = torch.from_numpy(masks)

        num_obj = min(len(logits), num_boxes)
        for obj_ind in range(num_obj):
            # box = boxes[obj_ind]

            m = masks[obj_ind][0]
            final_m += m
    final_m = (final_m > 0).to('cpu').numpy()
    # return np.dstack((final_m, final_m, final_m)) * 255
    return final_m > 0
