import torch
from vissl.data.collators import register_collator

# Temporary imports
# import debug_func


@register_collator("dense_collator")
def dense_collator(batch, create_multidimensional_tensor: bool = True):
    """
    This collator is used in 'dense' approach.

    The collator collates the batch for the following input (assuming k-views of iamge):

    Based on the 'multicrop_collator' implementation.

    How to crop 'common region' from views:

        view = data[view_idx][img_idx]
        tl, br = label[view_idx*img_N + img_idx]
        x_tl, y_tl = tl
        x_br, y_br = br

        view_crop = view[:, y_tl:y_br, x_tl:x_br]

    Args:
        batch (list): List of dicts.
            batch = [ 
                (0) img 1:   {"data": [img1_view1, ..., img1_viewM],
                              "data_valid": [img1_val1, ..., img1_valM],
                              "data_idx": [idx1, ..., idx1],
                              "label": tensor (k, 2, 2)},
                (1) img 2:   {...},
                ...
                (N-1) img N: {...}
            ]

    Returns:
        output_batch (dict): One batch of img_N*view_M samples.
            'data' (list): List of list with #views of torch.Tensor views.
                           [
                               [
                                   view_1 (img_N, 3, h, w),
                                   ... ,
                                   view_M (img_N, 3, h, w)
                               ]
                           ]
            'label' (list): List with single torch.Tensor (#views*#imgs, 2, 2)
            "data_valid": torch.tensor(sample_idx).
            "data_idx": torch.tensor(sample_idx).
    """
    assert "data" in batch[0], "data not found in sample"
    assert "label" in batch[0], "label not found in sample"
    assert "superpixels" in batch[0], "superpixels not found in sample"

    # Image-ordered list of lists [img_idx] --> [view_1, ... , view_M]
    #     Indexing: data[img_idx][view_idx] --> data for view 'i' for img 'j'
    #     data = [ [view_1, ... , view_5]_img_1, [view_1, ... , view_5]_img_2 ]
    data = [x["data"] for x in batch]
    labels = [torch.tensor(x["label"]) for x in batch]
    superpixels = [x["superpixels"] for x in batch]
    data_valid = [torch.tensor(x["data_valid"]) for x in batch]
    data_idx = [torch.tensor(x["data_idx"]) for x in batch]
    num_views, num_images = len(data[0]), len(data)

    # View-ordered list of lists [view_idx] --> [img_1, ... , img_N]
    #
    # data:
    #     Indexing: [ [img_1, ..., img_N]_view_1, ... , [img_1, ..., img_N]_view_M]
    #     data = [ [img_1, img_2]_view_1, ... , [img_1, img_2]_view_4]
    # label:
    #     Indexing: label[img_idx + img_N*view_idx] --> label for view 'i' for img 'j'
    #     label: [view_1_img_1, view_1_img_2, ..., view_M_img_N]
    output_data = []
    output_label = []
    output_superpixels = []
    output_data_valid = []
    output_data_idx = []
    for view_idx in range(num_views):
        _view_data = []
        _view_superpixels = []
        for img_idx in range(num_images):
            _view_data.append(data[img_idx][view_idx])
            _view_superpixels.append(superpixels[img_idx][view_idx])
            output_label.append(labels[img_idx][view_idx])
            output_data_valid.append(data_valid[img_idx][view_idx])
            output_data_idx.append(data_idx[img_idx][view_idx])
        output_data.append(torch.stack(_view_data))
        output_superpixels.append(torch.stack(_view_superpixels))

    # if create_multidimensional_tensor:
    #     output_data = MultiDimensionalTensor.from_tensors(output_data)
    output_batch = {
        "data": [output_data],
        "superpixels": [output_superpixels],
        "label": [torch.stack(output_label)],
        "data_valid": [torch.stack(output_data_valid)],
        "data_idx": [torch.stack(output_data_idx)],
    }

    # debug_func.plot_dense_collator_output_batch(output_batch)

    return output_batch
