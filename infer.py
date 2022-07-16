import cv2
import os
import numpy as np

import torch

from u2net import U2NET 


def norm_pred(d):
    """
    normalize prediction result

    Args:
        d (torch.tensor): tensor output

    Returns:
        _type_: normalized output
    """
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def resize(image, dst_size):
    """
    resize image to dst size with 
    INTER_LINEAR as default interpolation method 

    Args:
        image (np.array): image array
        dst_size (tuple(int, int)): destination size

    Returns:
        np.array: resized image array
    """
    return cv2.resize(image, dst_size, interpolation=cv2.INTER_LINEAR)


def normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    normalize image from [0,255] to [-1,1]

    Args:
        image (np.array): image array
        mean (list, optional): normalize mean. Defaults to [0.485, 0.456, 0.406].
        std (list, optional): normalize std. Defaults to [0.229, 0.224, 0.225].

    Returns:
        _type_: _description_
    """
    
    image = image / 255.0
    image = (image - np.array(mean)) / np.array(std)

    return image


def pre_resize(image, long_side=768):
    """_summary_

    Args:
        image (_type_): _description_
        long_side (int, optional): _description_. Defaults to 768.

    Returns:
        _type_: _description_
    """
    h, w, c = image.shape
    if max(h, w) > long_side:
        if h > w:
            new_h = long_side
            new_w = new_h / h * w
        else:
            new_w = long_side
            new_h = new_w / w * h 
        image = resize(image, (int(new_w), int(new_h)))
    
    return image
    


def preprocess(image, input_size=(320, 320)):
    """
    preprocess an image to torch tensor

    Args:
        image (np.array): input image array
        input_size (tuple(int, int)): input_size which is used during train and infer

    Returns:
        torch.Tensor: input image tensor
    """
    image = resize(image, input_size)
    image = normalize(image)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return torch.from_numpy(image)


def load_model(model_path):
    """load cnn model

    Args:
        model_path (str): path to trained cnn model

    Raises:
        Exception: ckpt not exists exception

    Returns:
        nn.Module: cnn model with weight loaded
    """

    
    net = U2NET(3, 1)
    print("...load U2NET---173.6 MB")
    
    if not os.path.exists(model_path):
        raise Exception("model checkpoint file does not exists")
    
    ckpt = torch.load(model_path, map_location='cpu')
    net.load_state_dict(ckpt['state_dict']) 
    
    if torch.cuda.is_available():
        net.cuda()

    net.eval()
    return net


def postprocess_composed(predict, image_org):
    """
    merge predicted mask and original image together

    Args:
        predict (np.array): predicted mask from cnn model
        image_org (np.array): original input image 

    Returns:
        np.array: composed image with white bg
    """
    
    if predict.ndim == 2:
        predict = predict[..., np.newaxis]

    composed = predict * image_org +\
         (1.0 - predict)  * np.array([255, 255, 255])
    return composed


def erode_and_dilate(mask):
    #TODO whether to set threshold???
    mask[mask < 30 / 255] = 0
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def save_output(image_name, img_org, pred, save_dir):
    """
    post processing and result save

    Args:
        image_name (str): input image name
        img_org (np.array): original input image 
        pred (torch.Tensor): prediction of cnn model
        save_dir (str): path to directory to save result
    """
    
    org_shape = img_org.shape[:2]
    predict_np = pred.squeeze().cpu().data.numpy()
    
    #TODO post process for mask
    predict_np = erode_and_dilate(predict_np)
    
    predict_np = cv2.resize(predict_np, (
        org_shape[1], org_shape[0]), interpolation=cv2.INTER_AREA)

    composed = postprocess_composed(predict_np, img_org).astype(np.uint8)

    img_name = os.path.splitext(os.path.basename(image_name))[0]
    mask_save_path = os.path.join(save_dir, img_name+"_mask.png")
    composed_save_path = os.path.join(save_dir, img_name+"_composed.png")
    
    cv2.imwrite(mask_save_path, (predict_np * 255).astype(np.uint8))
    cv2.imwrite(composed_save_path, composed)
    
    return mask_save_path, composed_save_path 
    

def remove_bg(img_path, model, save_dir):
    """
    remove background for image

    Args:
        img_path (str): path of image which needs to be background-removed
        model (nn.Module): cnn model
        save_dir (str): directory to save mask and composed image

    Returns:
        tuple(str, str): result composed image and its mask
    """
    if img_path is None:
        raise Exception("image path is None")
    if not os.path.exists(img_path):
        raise Exception("image path does not exists") 
    
    image_name = os.path.splitext(os.path.basename(img_path))[0]
    img_arr = cv2.imread(img_path, -1).astype(np.float32)
    
    if img_arr.ndim == 2:
        img_arr = np.expand_dims(img_arr, axis=-1)
        img_arr = np.stack([img_arr] * 3, axis=-1)
    else:
        h, w, c = img_arr.shape
        if c == 1:
           img_arr = np.stack([img_arr] * 3, axis=-1)
        elif c == 4:
            img_arr = img_arr[..., :3] 
    
    img_arr = pre_resize(img_arr)
    img_tensor = preprocess(img_arr)
    img_tensor = img_tensor.type(torch.FloatTensor)
    img_tensor = img_tensor.cuda()

    d0 = model(img_tensor)[0]
    pred = d0[:, 0, :, :]
    pred = norm_pred(pred)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    mask_path, composed_path = save_output(image_name, img_arr, pred, save_dir)
    
    return mask_path, composed_path


def demo():
    img_path = "./images/lihua.jpg"
    model = load_model("./checkpoint/u2net_furiends_v1_0716.pth")
    remove_bg(img_path, model, "./images")

if __name__ == "__main__":
    demo()