import torch
import torch.nn.functional as F


def resize_and_pad_tensor(tensor, target_size):
    _, _, h, w = tensor.shape

    # Calculate the scaling factor while preserving the aspect ratio
    scale_factor = min(target_size[0] / h, target_size[1] / w)
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)

    # Resize the tensor
    resized_tensor = F.interpolate(
        tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
    )

    # Pad the tensor to the target size
    pad_h = target_size[0] - new_h
    pad_w = target_size[1] - new_w
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)  # (left, right, top, bottom)

    padded_tensor = F.pad(resized_tensor, padding, mode="constant", value=0)

    return padded_tensor


def adjust_dimension(min_dim, max_dim, max_allowed):
    dim_length = max_dim - min_dim
    adjustment = (16 - dim_length % 16) % 16
    min_dim -= adjustment // 2
    max_dim += adjustment - adjustment // 2

    min_dim = max(min_dim, 0)
    max_dim = min(max_dim, max_allowed)

    if max_dim - min_dim != dim_length + adjustment:
        adjustment = (16 - (max_dim - min_dim) % 16) % 16
        if min_dim == 0:
            max_dim += adjustment
        elif max_dim == max_allowed:
            min_dim -= adjustment
        else:
            max_dim += adjustment // 2
            min_dim -= adjustment - adjustment // 2

    return max(min_dim, 0), min(max_dim, max_allowed)


def crop_to_mask_tensor(image_tensor, mask):
    """
    Crop a 3D image tensor to the region specified by the mask.
    ! ensure the dimensions of the cropped area are multiples of 16.

    Parameters:
    - image_tensor : torch.Tensor
        A 3D tensor representing the image with shape (B, C, H, W).
        If the image shape is (B, H, W, C), try to use image_tensor.permute(0, 3, 1, 2)
    - mask : torch.Tensor
        A 2D tensor representing the mask with shape (1, H, W), containing True for
        pixels to keep and False for the background.

    Returns:
    - torch.Tensor
        Cropped image tensor.
    """
    if image_tensor.shape[3] == 3:
        image_tensor = image_tensor.permute(0, 3, 1, 2)
    if image_tensor.size(2) != mask.size(1) or image_tensor.size(3) != mask.size(2):
        raise ValueError("The size of the mask must match the height and width of the image tensor")

    target_size = image_tensor.size(2), image_tensor.size(3)
    cropped_tensors = []
    for b in range(image_tensor.size(0)):
        img = image_tensor[b]
        msk = mask[b]

        non_background_indices = torch.nonzero(msk.squeeze(0), as_tuple=False)
        min_height = torch.min(non_background_indices[:, 0]).item()
        max_height = torch.max(non_background_indices[:, 0]).item() + 1
        min_width = torch.min(non_background_indices[:, 1]).item()
        max_width = torch.max(non_background_indices[:, 1]).item() + 1

        min_height, max_height = adjust_dimension(min_height, max_height, msk.size(0))
        min_width, max_width = adjust_dimension(min_width, max_width, msk.size(1))

        cropped_img = img[..., min_height:max_height, min_width:max_width]
        # padded_img = pad_tensor(cropped_img.unsqueeze(0), target_size)
        padded_img = resize_and_pad_tensor(cropped_img.unsqueeze(0), target_size)
        cropped_tensors.append(padded_img)

    cropped_tensor = torch.cat(cropped_tensors, dim=0)
    return cropped_tensor
