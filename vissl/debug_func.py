from random import sample
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import copy
import torch


def plot_img(img):

    plt.imshow(img)
    plt.show()


def plot_tensor(tensor, idx=None):

    if idx:
        tensor = tensor[idx].cpu().numpy()

    img = tensor.cpu().numpy()
    img = np.moveaxis(img, 0, -1)

    plt.imshow(img)
    plt.show()


def visualize_item(item):
    """
    """
    views = item['data']
    labels = item['labels']

    N = len(views)

    for idx in range(N):
        view_img = views[idx]
        x_tl, y_tl = labels[idx][0]
        x_br, y_br = labels[idx][1]
        box = [x_tl, y_tl, x_br, y_br]
        draw = ImageDraw.Draw(view_img)
        draw.rectangle(box, outline="blue", width=2)

        plt.subplot(1, N, idx + 1)  # Index must start from 1
        plt.imshow(view_img)
        plt.title(f"view {idx}")

    plt.show()


def plot_img_views(img, view_centers, common_rectangle, h, w):
    """
    Args:
        img (PIL image): Parent image of views.
        view_centers (list): View center coordinate pairs [(x,y)_1, ... , (x,y)_N]
        h (int): View pixel height.
        w (int): View pixel width.
    """

    common_x0 = common_rectangle[0][0]
    common_y0 = common_rectangle[0][1]
    common_x1 = common_rectangle[1][0]
    common_y1 = common_rectangle[1][1]

    draw = ImageDraw.Draw(img)
    draw.rectangle([common_x0, common_y0, common_x1, common_y1],
                   outline="red",
                   width=5)

    ################################
    #  Plot views on parent image
    ################################

    img2 = copy.deepcopy(img)
    draw = ImageDraw.Draw(img2)

    N = len(view_centers)
    for idx in range(N):
        x_center, y_center = view_centers[idx]

        x_0 = int(x_center - 0.5 * h)
        y_0 = int(y_center - 0.5 * w)

        x_1 = int(x_center + 0.5 * h)
        y_1 = int(y_center + 0.5 * w)

        draw.rectangle([x_0, y_0, x_1, y_1], outline="blue", width=2)

    plt.imshow(img2)
    plt.show()

    ##############################################
    #  Plot individual views with common region
    ##############################################

    for idx in range(N):

        x_center, y_center = view_centers[idx]

        x_0 = int(x_center - 0.5 * h)
        y_0 = int(y_center - 0.5 * w)
        x_1 = int(x_center + 0.5 * h)
        y_1 = int(y_center + 0.5 * w)

        plt.subplot(1, N, idx + 1)  # Index must start from 1
        box = (x_0, y_0, x_1, y_1)
        plt.imshow(img.crop(box))
        plt.title(f"view {idx}")

    plt.show()


def plot_view_and_label(view_img, label):
    """
    Args:
        view_img (PIL Image):
        label (list): List of top-left and bottom-right coordinate pairs
                      constituting the common region rectangle in view coord.
                      [ (x, y)_tl, (x, y)_br ]
    """

    # Non-negate flipped labels
    if label[0][0] < 0:
        label = np.negative(label).tolist()
    # x_tl, y_tl, x_br, y_br
    box = [label[0][0], label[0][1], label[1][0], label[1][1]]
    draw = ImageDraw.Draw(view_img)
    draw.rectangle(box, outline="blue", width=2)
    plt.imshow(view_img)
    plt.show()


def plot_common_region_crops(views, labels):
    """
    views (list): List of PIL.Image objects.
    labels (list): List of pairs of coordinate tuples
    """
    M = len(views)
    for idx in range(M):
        view = torch.Tensor(np.array(views[idx])).int()
        tl, br = labels[idx]
        x_tl, y_tl = tl
        x_br, y_br = br

        view = view[y_tl:y_br, x_tl:x_br, :]

        plt.subplot(1, M, idx + 1)
        plt.imshow(view)

    plt.show()


def plot_dataset_item(item):
    """
    Visualize the final 'item' sample as returned from 'dense_ssl_dataset' to
    the dataloader. Also visualizes the resulting 'common region' crop using
    the label, and the edge maps for each view.

    NOTE: Disable appearance augmentations and normalization for visual
          comparison.

    Args:
        item (dict):
            'views' (list): List of views as torch.Tensor (3, h, w).
            'label' (list): List of crop box coordinate pairs (#views, 2, 2)
                            [ ( [x_tl, y_tl], [x_br, y_br] ), ... ].
            'superpixels' (list): List of view superpixels as torch.Tensor
                                  (h, w).
    """
    views = item['data']
    labels = item['label']
    superpixels = item['superpixels']

    M = len(views)
    for idx in range(M):
        view = views[idx]
        # Non-negate flipped labels
        if torch.any(torch.tensor(labels[idx]) < 0):
            labels[idx] = np.negative(labels[idx]).tolist()
        tl, br = labels[idx]
        x_tl, y_tl = tl
        x_br, y_br = br

        view_crop = view[:, y_tl:y_br, x_tl:x_br]

        superpixel = superpixels[idx]
        superpixel_crop = superpixel[y_tl:y_br, x_tl:x_br]

        plt.subplot(4, M, idx + 1)
        plt.imshow(np.transpose(view.numpy(), (1, 2, 0)))

        plt.subplot(4, M, M + (idx + 1))
        plt.imshow(superpixel.numpy())

        plt.subplot(4, M, 2 * M + (idx + 1))
        plt.imshow(np.transpose(view_crop.numpy(), (1, 2, 0)))

        plt.subplot(4, M, 3 * M + (idx + 1))
        plt.imshow(superpixel_crop.numpy())

    plt.show()


def plot_dense_collator_output_batch(output_batch):
    """
    Args:
        output_batch (dict):
            'data' (list): List of list with #views of torch.Tensor views.
                           [ [view_1 (img_N, 3, h, w),
                             ... ,
                             view_M (img_N, 3, h, w)]]
            'label' (list): List with single torch.Tensor (#views*#imgs, 2, 2)
    """
    data = output_batch['data'][0]
    label = output_batch['label'][0]
    edges = output_batch['edges'][0]

    img_N = data[0].shape[0]
    view_M = len(data)

    for img_idx in range(img_N):
        for view_idx in range(view_M):

            view = data[view_idx][img_idx]
            edge = edges[view_idx][img_idx]
            tl, br = label[view_idx * img_N + img_idx]
            x_tl, y_tl = tl
            x_br, y_br = br

            view_crop = view[:, y_tl:y_br, x_tl:x_br]

            print(3 * img_idx * view_M + view_idx + 1,
                  (3 * img_idx + 1) * view_M + view_idx + 1,
                  (3 * img_idx + 2) * view_M + view_idx + 1)

            plt.subplot(3 * img_N, view_M, 3 * img_idx * view_M + view_idx + 1)
            plt.imshow(np.transpose(view.numpy(), (1, 2, 0)))
            plt.subplot(3 * img_N, view_M,
                        (3 * img_idx + 1) * view_M + view_idx + 1)
            plt.imshow(np.transpose(view_crop.numpy(), (1, 2, 0)))
            plt.subplot(3 * img_N, view_M,
                        (3 * img_idx + 2) * view_M + view_idx + 1)
            plt.imshow((edge.numpy()))

    plt.show()


def plot_sample_for_model(sample):
    """
    """
    input = [x.cpu() for x in sample['input']]
    target = [x.cpu() for x in sample['target']]

    img_N = input[0].shape[0]
    view_M = len(input)

    for img_idx in range(img_N):
        for view_idx in range(view_M):

            view = input[view_idx][img_idx]
            tl, br = target[view_idx * img_N + img_idx]
            x_tl, y_tl = tl
            x_br, y_br = br

            view_crop = view[:, y_tl:y_br, x_tl:x_br]

            print(2 * img_idx * view_M + view_idx + 1,
                  (2 * img_idx + 1) * view_M + view_idx + 1)

            plt.subplot(2 * img_N, view_M, 2 * img_idx * view_M + view_idx + 1)
            plt.imshow(np.transpose(view.numpy(), (1, 2, 0)))
            plt.subplot(2 * img_N, view_M,
                        (2 * img_idx + 1) * view_M + view_idx + 1)
            plt.imshow(np.transpose(view_crop.numpy(), (1, 2, 0)))

    plt.show()


def plot_model_input_crops(input, target, img_N, view_M):
    """
    Indexing: [view_idx*img_N + img_idx]

    Args:
        input (torch.Tensor): Model input tensor (b, 3, h, w)
    """
    input = input.cpu()
    target = target.cpu()

    for img_idx in range(img_N):
        for view_idx in range(view_M):

            sample_idx = view_idx * img_N + img_idx

            view = input[sample_idx]

            # Non-negate flipped labels
            if torch.any(target[sample_idx] < 0):
                target[sample_idx] = torch.neg(target[sample_idx])
            tl, br = target[sample_idx]
            x_tl, y_tl = tl
            x_br, y_br = br

            view_crop = view[:, y_tl:y_br, x_tl:x_br]

            print(2 * img_idx * view_M + view_idx + 1,
                  (2 * img_idx + 1) * view_M + view_idx + 1)

            plt.subplot(2 * img_N, view_M, 2 * img_idx * view_M + view_idx + 1)
            plt.imshow(np.transpose(view.numpy(), (1, 2, 0)))
            plt.subplot(2 * img_N, view_M,
                        (2 * img_idx + 1) * view_M + view_idx + 1)
            plt.imshow(np.transpose(view_crop.numpy(), (1, 2, 0)))

    plt.show()
