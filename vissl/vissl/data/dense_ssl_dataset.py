import copy
from typing import Callable, Dict, Set

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from numpy.random import default_rng

# Temporary imports
# import debug_func
from vissl.config import AttrDict
from vissl.data.ssl_dataset import GenericSSLDataset

UNIFORM_SAMPLING_PROB = 0.005
AUG_TYPE_IDXS = [0, 1]
PADDING_COLOR = (0, 0, 0)


class DenseSSLDataset(GenericSSLDataset):
    """
    Dense Self Supervised Learning Dataset Class.

    NOTE: Usage of 'vanilla' or 'dense' dataloader is determined in
          vissl/data/__init__.py

    TODO: Confirm workings of view geometry transformations w. label coords.

    How to crop 'common region' from views:

        view (3, h, w) = views[idx]
        crop_tl (2), crop_br (2) = labels[idx]

        x_tl, y_tl = crop_tl
        x_br, y_br = crop_br

        common_crop = view[:, y_tl:y_br, x_tl:x_br]

    Args:
        cfg (AttrDict): configuration defined by user
        split (str): the dataset split for which we are constructing the
                     Dataset object.
        dataset_source_map (Dict[str, Callable]): The dictionary that maps
                    what data sources are supported and what object to use to
                    read data from those sources. For example:
                    DATASET_SOURCE_MAP = {
                        "disk_filelist": DiskImageDataset,
                        "disk_folder": DiskImageDataset,
                        "synthetic": SyntheticImageDataset,
                    }
        data_sources_with_subset (Set[str]): the set of datasets for which the
                    subset operation is supported inside GenericSSLDataset

    Returns:
        item (dict):
            'data' (list): List of views as torch.Tensor (3, h, w).
            'label' (list): List of crop box coordinate pairs (#views, 2, 2)
                            [ ( [x_tl, y_tl], [x_br, y_br] ), ... ].
    """

    def __init__(
        self,
        cfg: AttrDict,
        split: str,
        dataset_source_map: Dict[str, Callable],
        data_sources_with_subset: Set[str],
        **kwargs,
    ):
        super().__init__(cfg, split, dataset_source_map,
                         data_sources_with_subset)
        config = cfg.DATA.TRAIN.dense_swav
        # Standard image parameters
        self.std_img_w = config.std_image_width
        self.std_img_h = config.std_image_height
        self.max_img_size = 2048  # Sizes above will be resized
        if self.std_img_w > self.std_img_h:
            self.min_img_size = self.std_img_h  # Sizes bellow will be upsampled and padded
        else:
            self.min_img_size = self.std_img_w
        self.std_img_aspect_ratio = self.std_img_w / self.std_img_h
        # Set view sampling parameters
        self.view_h = config.view_size
        self.view_w = config.view_size
        self.view_N = config.view_n
        self.fpn_center_crop = config.fpn_center_crop
        self.superpixel_region_size = config.superpixel_region_size
        self.superpixel_ruler = config.superpixel_ruler
        self.superpixel_iters = config.superpixel_iters
        self.regular_grid = config.regular_grid
        self.mask_intensity = config.mask_intensity
        self.common_idx_preserve_ratio = config.common_idx_preserve_ratio
        self.max_common_N = config.max_common_size
        self.view_R = int(self.view_h // 2) - 1  # Maximum translation
        # Concentration parameters for Von Mises distributions
        self.k_theta = config.view_sampling_conc_param_theta
        self.k_r = config.view_sampling_conc_param_r
        # Geometric augmentations
        self.flip_views = config.flip_views
        self.resize_range = config.resize_range
        # Resolution augmentation
        self.res_ratios = config.res_ratios
        # Augmentation types
        self.prob_res_aug = config.prob_res_aug
        self.prob_con_aug = 1. - self.prob_res_aug
        self.AUG_TYPE_PROBS = [self.prob_con_aug, self.prob_res_aug]

        # Run debug function to analyze number of expected elements in common
        # region
        count_common_view_elem = cfg.DATA.analyze_common_view_element_count
        if count_common_view_elem:
            img = PIL.Image.new('RGB', (self.std_img_w, self.std_img_h))
            self.analyze_common_view_elem_count(img, count_common_view_elem)
            exit()

        assert self.view_N >= 2, f"Views must be >= 2 (now {self.view_N})"

    def generate_sample(self, idx):
        """Generates a sample for image specified by 'idx' using randomly
        sampled parameters.

        Args:
            idx (int): Image index in file list used to generate views.

        Returns:
            views [PIL.Image]: List of RGB view images of standard size.
            labels [list]: List of lists representing common region in view
                           coordinates [[x, y]_tl, [x, y]_br].
            superpixels [np.array]:  List of arrays storing the 'common idx'
                                     superpixel regions for the image in the
                                     view reference frame.
            noise_masks [PIL.Image]: Boolean masks for each view representing
                                     common regions for all views.
        """
        if not self._labels_init and len(self.label_sources) > 0:
            self._load_labels()
            self._labels_init = True

        subset_idx = idx
        if self.data_limit >= 0 and self._can_random_subset_data_sources():
            if not self._subset_initialized:
                self._init_image_and_label_subset()
            subset_idx = self.image_and_label_subset[idx]

        # TODO: this doesn't yet handle the case where the length of datasets
        # could be different.
        item = {"data": [], "data_valid": [], "data_idx": []}
        for data_source in self.data_objs:
            data, valid = data_source[subset_idx]
            item["data"].append(data)
            # Add values for all views
            item["data_idx"] += [idx] * self.view_N
            item["data_valid"] += [1 if valid else -1] * self.view_N
        # item['data'] : [img]

        #############################
        #  Global image transforms
        #############################
        # Target 1280 x 960 resolution
        img = item['data'][0]  # PIL.Image
        img = self.standardize_img_size(img)

        ###############################
        #  View and label generation
        ###############################

        views, labels, superpixels = self.gen_views(img)

        views, superpixels, noise_masks = self.mask_views(
            views, labels, superpixels)

        views, labels, superpixels, noise_masks = self.standardize_view_size(
            views, labels, superpixels, noise_masks, self.view_h)

        superpixels = self.remove_noncommon_region_idxs(superpixels, labels)

        return item, views, labels, superpixels, noise_masks

    def __getitem__(self, idx: int):
        """
        """
        # Ensure that randomly generated sample contains learnable regions
        while True:
            sample = self.generate_sample(idx)
            item, views, labels, superpixels, noise_masks = sample
            # Remove default value (-1) from count
            learnable_region_count = len(np.unique(superpixels)) - 1
            if learnable_region_count > 0:
                break
            else:
                # Randomly sample new 'idx' and redo generation
                idx = np.random.randint(0, idx)

        # print("\ndataset")
        # for idx in range(len(superpixels)):
        #     superpixel = superpixels[idx]
        #     idx_set = np.unique(superpixel)
        #     print(idx, len(idx_set))

        ##############################################################
        #  View geometric augmentation
        #  NOTE: Appearance agumentation done by vanilla transforms
        ##############################################################
        # TODO
        views, labels, superpixels, noise_masks = self.transform_view_geom(
            views, labels, superpixels, noise_masks, self.flip_views,
            self.resize_range)

        item['data'] = views
        item['label'] = labels
        item['superpixels'] = superpixels

        # apply the transforms on the image
        if self.transform:
            item = self.transform(item)
        # item['data'] : [ [img_aug1, img_aug2] ]

        # Apply masks after appearance augmentations
        views = item['data']
        views = self.apply_masks(views, noise_masks)
        item['data'] = views

        # Convert to torch.Tensor as done for 'views' in transform func
        item['superpixels'] = [
            torch.from_numpy(a.copy()) for a in item['superpixels']
        ]

        # debug_func.plot_dataset_item(item)
        # exit()

        return item

    @staticmethod
    def apply_masks(views, masks):
        """Applies noise masks on a set of views.

        Args:
            views (list): View images as torch.Tensor w. dim (h, w, 3).
            masks (list): Boolean np.array mask matrices representing view
                          regions to be masked out by noise.

        Returns:
            List of masked view images as torch.Tensor.
        """
        # Process each view one-by-one
        for idx in range(len(views)):
            _, view_size, _ = views[idx].shape
            # Generate noise matrix
            noise = 2. * np.random.rand(
                3,
                view_size,
                view_size,
            ) - 1.
            noise = torch.from_numpy(noise).float()
            # Prepare mask matrix
            mask = masks[idx]
            mask = np.array(mask, bool)
            mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(0).repeat(3, 1, 1)
            # Replace masked indices with noise values
            view = views[idx]
            view[mask] = noise[mask]
            views[idx] = view

        return views

    @staticmethod
    def remove_noncommon_region_idxs(superpixels, labels):
        """Finds the set of common region idxs and removes uncommon ones from
        the superpixel idx maps.
        """
        # Ensure all region idx in crops are common
        common_idx_set = np.unique(superpixels[0])
        common_idx_set = list(common_idx_set)
        common_idx_set = set(common_idx_set)

        for idx in range(1, len(superpixels)):
            label = labels[idx]
            if label[0][0] < 0:
                label = -1 * label

            tl, br = labels[idx]
            x_tl, y_tl = tl
            x_br, y_br = br
            superpixel = superpixels[idx]
            superpixel = superpixel[y_tl:y_br, x_tl:x_br]

            idx_set = np.unique(superpixel)
            idx_set = list(idx_set)
            idx_set = set(idx_set)

            common_idx_set = common_idx_set.intersection(idx_set)

        for idx in range(len(superpixels)):
            superpixel = superpixels[idx]
            idx_set = np.unique(superpixel)
            idx_set = list(idx_set)
            idx_set = set(idx_set)
            noncommon_idx_set = idx_set - common_idx_set

            for noncommon_idx in noncommon_idx_set:
                superpixel[superpixel == noncommon_idx] = -1
            superpixels[idx] = superpixel

        return superpixels

    def standardize_img_size(self, img):
        """
        Returns a resized image with the smallest dimension equal to the
        standard image dimension.

        TODO: Refactor upper and lower bound to single flow

        Args:
            img (PIL.Image)

        Returns:
            img (PIL.Image)
        """
        w, h = img.size

        # 1) Upsample larger side to minimum image size
        if w > h and w < self.min_img_size:
            scaling = self.min_img_size / w
            w_ = int(scaling * w)
            h_ = int(scaling * h)
            img = img.resize((w_, h_), PIL.Image.BICUBIC)
            w, h = img.size

        elif h > w and h < self.min_img_size:
            scaling = self.min_img_size / h
            w_ = int(scaling * w)
            h_ = int(scaling * h)
            img = img.resize((w_, h_), PIL.Image.BICUBIC)
            w, h = img.size

        # 2) Add center padding to smaller side equaling minimum image size
        if w < self.min_img_size:

            diff = self.min_img_size - w
            img_padding = PIL.Image.new(img.mode, (self.min_img_size, h),
                                        PADDING_COLOR)
            img_padding.paste(img, (diff // 2, 0))
            img = img_padding

        elif h < self.min_img_size:

            diff = self.min_img_size - h
            img_padding = PIL.Image.new(img.mode, (w, self.min_img_size),
                                        PADDING_COLOR)
            img_padding.paste(img, (0, diff // 2))
            img = img_padding

        # 3) Downsample image so all sides are smaller than maximum image size
        if w > self.max_img_size or h > self.max_img_size:
            aspect_ratio = w / h

            if aspect_ratio <= self.std_img_aspect_ratio:
                scaling = self.max_img_size / h

            elif aspect_ratio > self.std_img_aspect_ratio:
                scaling = self.max_img_size / w

            w_ = int(scaling * w)
            h_ = int(scaling * h)
            img = img.resize((w_, h_), PIL.Image.BICUBIC)

        # Only standardize too small images <-- OBSOLOTE ?
        # if w > self.std_img_w and h > self.std_img_h:
        #     return img
        # aspect_ratio = w / h
        # if aspect_ratio <= self.std_img_aspect_ratio:
        #     scaling = self.std_img_w / w
        # elif aspect_ratio > self.std_img_aspect_ratio:
        #     scaling = self.std_img_h / h
        # w_ = int(scaling * w)
        # h_ = int(scaling * h)
        # img = img.resize((w_, h_), PIL.Image.BICUBIC)

        if self.fpn_center_crop:
            w_, h_ = img.size

            if w_ > h_:
                smallest_dim = h_
            else:
                smallest_dim = w_

            left = (w_ - smallest_dim) / 2
            top = (h_ - smallest_dim) / 2
            right = (w_ + smallest_dim) / 2
            bottom = (h_ + smallest_dim) / 2

            # Crop the center of the image
            img = img.crop((left, top, right, bottom))

            # Resize to standard image size
            img = img.resize((self.std_img_w, self.std_img_h),
                             PIL.Image.BICUBIC)

        return img

    def gen_views(self, img):
        """

        Args:
            img (PIL Image):
            #item (dict): Key 'data' --> [img].

        Returns:
            Tuple: [0] List of view images (PIL Image).
                   [1] List of common region rectangle top-left, bottom-right
                       coordinates [(x, y)_tl, (x, y)_br].
                   [2] List of view edge masks (np.array)
            #Same 'item' dict with key 'data' --> [ [view_1, ..., view_N] ].
        """
        # Generate set of view centers and common region coordinates in parent
        # image coordiantes
        # view_centers: [ (x, y)_1, ... , (x, y)_N ]
        # common_rectangle: [ (x, y)_tl, (x, y)_br ]
        view_centers, view_sizes = self.gen_view_centers(img)
        common_rectangle = self.common_rectangle(view_centers, view_sizes)

        # Prevent memory overshooting by thresholding maximum size
        common_rectangle = self.threshold_common_region_size(common_rectangle)

        # debug_func.plot_img_views(
        #    img, view_centers, common_rectangle, self.view_h, self.view_w)

        ###########
        #  Edges
        ###########
        # low_threshold = 15
        # ratio = 3
        # kernel_size = 3
        # sigma = 0.33

        # img_gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
        # img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

        # v = np.median(img_blur)
        # low_threshold = int(max(0, (1.0 - sigma) * v))
        # high_threshold = int(min(255, (1.0 + sigma) * v))

        # detected_edges = cv2.Canny(img_blur,
        #                            low_threshold,
        #                            high_threshold,
        #                            L2gradient=True)
        # edge_mask = detected_edges != 0

        # plt.subplot(1, 2, 1)
        # plt.imshow(img_gray)
        # plt.subplot(1, 2, 2)
        # plt.imshow(edge_mask)
        # plt.show()

        ##########
        #  Grid
        ##########
        if self.regular_grid:
            W, H = img.size
            num_px = H * W
            num_elem = int(num_px / self.superpixel_region_size**2)

            img_superpixels = np.arange(0, num_elem, 1, dtype=np.int32)

            # For comparing pixel-only results
            if self.superpixel_region_size == 1:
                img_superpixels = img_superpixels.reshape(H, W)
            else:
                # How to compute rescaling coefficient 'a' for obtaining target
                # element count:
                #     H * W = N
                #     (a*H) * (a*W) = a^2 * N
                #       H'  *   W'  =    N'
                #         ==> a = sqrt(N' / N)
                a = np.sqrt(num_elem / num_px)
                W_elem = int(a * W)
                H_elem = int(a * H)

                img_superpixels = np.arange(0,
                                            W_elem * H_elem,
                                            1,
                                            dtype=np.int32)
                img_superpixels = img_superpixels.reshape(-1, W_elem)

                img_superpixels = cv2.resize(img_superpixels, (W, H),
                                             interpolation=cv2.INTER_NEAREST)
        #################
        #  Superpixels
        #################
        else:
            cv2.setNumThreads(1)
            img_blur = np.asarray(img)[:, :, ::-1]
            img_blur = cv2.GaussianBlur(img_blur, (3, 3), cv2.BORDER_DEFAULT)
            slic = cv2.ximgproc.createSuperpixelSLIC(
                img_blur,
                algorithm=cv2.ximgproc.SLIC,
                region_size=self.superpixel_region_size,
                ruler=self.superpixel_ruler)
            slic.iterate(self.superpixel_iters)
            img_superpixels = slic.getLabels()  # np.array w. dim (H, W)

        # Generate set of views and common region labels in view image coords.
        views = []
        labels = []
        superpixels = []
        for idx in range(self.view_N):
            gen_view_tuple = self.gen_view_and_label(img, view_centers[idx],
                                                     view_sizes[idx],
                                                     common_rectangle,
                                                     img_superpixels)
            view_img, view_common_region, view_superpixels = gen_view_tuple
            views.append(view_img)
            labels.append(view_common_region)
            superpixels.append(view_superpixels)

        # debug_func.plot_common_region_crops(views, labels)
        return views, labels, superpixels

    def mask_views(self, views, labels, superpixels):
        """
        Returns:
            views [np.array]: List of arrays representing the input image view.
            superpixels [np.array]: List of arrays storing the 'common idx'
                                    superpixel regions for the image in the
                                    view reference frame.
        """
        # from PIL import ImageDraw
        rng = default_rng()

        # Compute set of common superpixels
        label = labels[0]
        # Non-negate flipped labels
        if torch.any(torch.tensor(label) < 0):
            label = np.negative(labels[0]).tolist()
        tl, br = label
        x_tl, y_tl = tl
        x_br, y_br = br

        superpixels_common_crop = superpixels[0][y_tl:y_br, x_tl:x_br]
        common_idxs = np.unique(superpixels_common_crop)

        # Grow a random subset until >50% of the mask is filled
        N = int(self.common_idx_preserve_ratio * len(common_idxs))
        preserved_common_idxs = np.random.choice(common_idxs, N, replace=False)

        # Mask each view one-by-one
        view_center_x = int(self.view_w / 2)
        view_center_y = int(self.view_h / 2)

        R = view_center_x if view_center_x <= view_center_y else view_center_y

        masks = []

        for view_idx in range(len(views)):

            view_size, _ = views[view_idx].size
            mask_w_max = self.mask_intensity * view_size
            mask_h_max = self.mask_intensity * view_size

            # Sample angle
            ang = rng.uniform(0, 2 * np.pi)
            # Sample radius with highest likelihood at 0.75 distance
            r = rng.vonmises(0.5 * np.pi, 2)  # [-pi, pi]
            r = r + np.pi  # [0, 2*pi]
            r = r * R / (2 * np.pi)  # [0, R]

            mask_x = r * np.cos(ang)
            mask_y = r * np.sin(ang)

            mask_x = int(mask_x)
            mask_y = int(mask_y)

            # Transform relative --> absolute coordinates
            mask_x += view_center_x
            mask_y += view_center_y

            # Sample mask width
            mask_w = rng.vonmises(0.5 * np.pi, 2)
            mask_w = mask_w + np.pi
            mask_w = mask_w * mask_w_max / (2 * np.pi)
            mask_w = int(mask_w)
            # Sample mask height
            mask_h = rng.vonmises(0.5 * np.pi, 2)
            mask_h = mask_h + np.pi
            mask_h = mask_h * mask_h_max / (2 * np.pi)
            mask_h = int(mask_h)

            # Clip coordinates to view
            x_min_margin = int(np.ceil(0.5 * mask_w))
            y_min_margin = int(np.ceil(0.5 * mask_h))
            x_max_margin = int(self.view_w - 0.5 * mask_w)
            y_max_margin = int(self.view_h - 0.5 * mask_h)

            if mask_x < x_min_margin:
                mask_x = x_min_margin
            elif mask_x > x_max_margin:
                mask_x = x_max_margin

            if mask_y < y_min_margin:
                mask_y = y_min_margin
            elif mask_y > y_max_margin:
                mask_y = y_max_margin

            mask_x_tl = mask_x - int(mask_w / 2)
            mask_y_tl = mask_y - int(mask_h / 2)
            mask_x_br = mask_x + int(mask_w / 2)
            mask_y_br = mask_y + int(mask_h / 2)

            # print(
            #     f"ang {ang}, mask_h {mask_h}, mask_w {mask_w}, mask_x \
            #       {mask_x}, mask_y {mask_y}"
            # )

            # draw = ImageDraw.Draw(views[view_idx])
            # draw.rectangle([mask_x_tl, mask_y_tl, mask_x_br, mask_y_br],
            #                outline="red",
            #                width=3)
            # plt.imshow(views[view_idx])
            # plt.show()

            # Compute set of masked superpixels
            superpixels_masked = superpixels[view_idx][mask_y_tl:mask_y_br,
                                                       mask_x_tl:mask_x_br]
            masked_idxs = np.unique(superpixels_masked)

            # Remove set of preserved common idxs
            masked_idxs = np.array(
                list(set(masked_idxs) - set(preserved_common_idxs)))

            # Remove masked common idxs
            common_idxs = np.array(list(set(common_idxs) - set(masked_idxs)))

            masks.append(masked_idxs)

        noise_masks = []
        for view_idx in range(len(views)):

            view_size, _ = views[view_idx].size

            superpixel_mask = np.isin(superpixels[view_idx],
                                      masks[view_idx],
                                      invert=False)

            # Create mask for later masking (after augmentations)
            mask = PIL.Image.fromarray(superpixel_mask)
            noise_masks.append(mask)
            # noise = np.random.randint(0,
            #                           256, (view_size, view_size, 3),
            #                           dtype=np.uint8)
            # noise = np.empty((view_size, view_size))
            # noise.fill(np.nan)
            # noise = PIL.Image.fromarray(noise)

            # views[view_idx] = PIL.Image.composite(
            #     views[view_idx], noise, mask)

            # plt.subplot(1, 2, 1)
            # plt.imshow(views[view_idx])
            # plt.subplot(1, 2, 2)
            # plt.imshow(superpixel_mask)
            # plt.show()

            # for view_idx in range(len(views)):
            sp = superpixels[view_idx]
            sp_label = -1 * np.ones(sp.shape, dtype=np.int32)

            for idx in list(common_idxs):
                sp_label[sp == idx] = idx

            superpixels[view_idx] = sp_label

        return views, superpixels, noise_masks

    @staticmethod
    def standardize_view_size(views, labels, superpixels, noise_masks, size):
        """
        """
        assert len(views) == len(superpixels)

        for idx in range(len(views)):
            view = views[idx]
            label = labels[idx]
            superpixel = superpixels[idx]
            noise_mask = noise_masks[idx]

            size_orig, _ = view.size

            view = view.resize((size, size), PIL.Image.BICUBIC)
            superpixel = cv2.resize(superpixel, (size, size),
                                    interpolation=cv2.INTER_NEAREST)
            noise_mask = noise_mask.resize((size, size), PIL.Image.NEAREST)

            # Scale all
            scaling_coef = size / size_orig
            label = np.array(label)
            label = np.floor(scaling_coef * label).astype(int)
            label = label.tolist()

            views[idx] = view
            labels[idx] = label
            superpixels[idx] = superpixel
            noise_masks[idx] = noise_mask

        return views, labels, superpixels, noise_masks

    def gen_view_centers(self, img):
        """
        TODO: Add 'size' generation to function
        TODO: Refactor different functions to separate functions

        Returns a list of view center coordinates centered around a location
        (x, y) within image spanned by (H, W).

        Internal view generation parameters <TODO Rewrite>
            view_h (int): View pixel height.
            view_w (int): View pixel width.
            view_N (int): Number of views to generate (incl. center view).
            view_R (float): Maximum view translation distance.
            k_theta (float): Concentration coef. for angle Von Mises sampling.
            k_r (float): Concetration coef. for distance Von Misese sampling.

        Args:
            H (int): Image pixel height.
            W (int): Image pixel width.

        Returns:
            List of view center coordinates [ [x, y]_1, ... ,  [x, y]_N ]
            List of view sizes [ s_1, ... , s_N ]
        """
        # Sample and add center view (x, y) with high probability to sample
        # edge points
        # Generate Canny edge map
        src = copy.deepcopy(np.asarray(img)[:, :, ::-1])
        src = cv2.GaussianBlur(src, (3, 3), 0)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        lower = 10
        upper = 50
        detected_edges = cv2.Canny(src, lower, upper, L2gradient=True)
        canny = detected_edges != 0

        canny = 255 * canny.astype(np.uint8)

        canny = cv2.dilate(canny, (3, 3), iterations=1)
        canny = cv2.GaussianBlur(canny, (3, 3), 0)

        # Change value range to probability values
        canny = canny.astype(np.float64) / 255.
        H, W = canny.shape

        # Add uniform probability to all elements
        canny += UNIFORM_SAMPLING_PROB

        # Normalize probability distribution
        canny = np.reshape(canny, (-1))
        canny = canny / np.sum(canny)

        # Sample center view point
        sampled_idx = np.random.choice(np.arange(0, len(canny)), p=canny)

        x = sampled_idx % W
        y = sampled_idx // W

        # Sample augmentation type
        aug_type = np.random.choice(AUG_TYPE_IDXS, p=self.AUG_TYPE_PROBS)

        # Contextual appearance augmentation
        if aug_type == 0:

            # Center view
            # Sample view size 's'
            rng = default_rng()
            a = rng.uniform()
            zoom_coef = self.resize_range[1] * a + self.resize_range[0] * (1. -
                                                                           a)
            center_s = zoom_coef * self.view_h

            x, y = self.clip_view_coordinates(x, y, H, W, center_s, center_s)
            view_centers = [(x, y)]
            view_sizes = [center_s]

            # Sample and add surrounding views
            for idx in range(self.view_N - 1):
                view_x, view_y, view_size = self.sample_view(
                    idx, self.view_N - 1, x, y, self.view_R, self.k_theta,
                    self.k_r, self.view_h, self.resize_range)

                # Keeps view within parent image
                view_x, view_y = self.clip_view_coordinates(
                    view_x, view_y, H, W, view_size, view_size)

                view_center = (view_x, view_y)
                view_centers.append(view_center)
                view_sizes.append(view_size)

        # Resolution appearance augmentation
        elif aug_type == 1:
            view_centers = []
            view_sizes = []

            for idx in range(self.view_N):
                # View center always same
                view_center = (x, y)
                view_centers.append(view_center)

                # View size differ
                rng = default_rng()
                p = rng.random()
                range_min = self.res_ratios[idx]
                range_max = self.res_ratios[idx + 1]
                ratio = (1. - p) * range_min + p * range_max
                # Ratio of default view size
                view_size = int(np.ceil(ratio * self.view_h))

                view_sizes.append(view_size)

        else:
            raise ValueError(f"Undefined augmentation type ({aug_type})")

        return view_centers, view_sizes

    def gen_view_and_label(self, img, view_center, view_size, common_region,
                           superpixels):
        """
        Args:
            img (PIL Image):      Parent RGB image.
            view_center (tuple):  View center coordinate pair (x, y).
            view_size (int):      View size in pixels.
            common_region (list): List of top-left and bottom-right coordinate
                                  pairs being the common region rectangle.
                                  [ [x, y]_tl, [x, y]_br ].
            superpixels (np.array):
        Returns:
            Tuple: [0] view image (PIL Image).
                   [1] common region in view coordinates [[x, y]_tl, [x, y]_br]
                   [2] superpixels (np.array).
        """
        ##########
        #  View
        ##########
        x, y = view_center
        # View rectangle top-left, bottom-right coordinates (x, y)
        view_tl, view_br = self.conv_center2rect(x, y, view_size, view_size)
        box = [view_tl[0], view_tl[1], view_br[0], view_br[1]]
        view_img = img.crop(box)

        ###########
        #  Label
        ###########
        # Convert 'common region' coordinates --> view coordinates
        view_x_tl, view_y_tl = view_tl[0], view_tl[1]
        # Common region in image frame coordinates
        x_tl, y_tl = common_region[0]
        x_br, y_br = common_region[1]
        # Common region in view coordinates
        x_tl, y_tl = self.transf_coord_img2view(x_tl, y_tl, view_x_tl,
                                                view_y_tl)
        x_br, y_br = self.transf_coord_img2view(x_br, y_br, view_x_tl,
                                                view_y_tl)

        view_common_region = [[x_tl, y_tl], [x_br, y_br]]

        ###########
        #  Edges
        ###########
        # First transform to 'view', then to 'common region'

        # Convert superpixel matrix into PIL Image to do padded cropping
        # NOTE: Need to add +1 to make 'ignore region' value 0
        view_superpixels = superpixels.astype(float) + 1.
        view_superpixels = PIL.Image.fromarray(view_superpixels)
        view_superpixels = view_superpixels.crop(
            (view_tl[0], view_tl[1], view_br[0], view_br[1]))

        view_superpixels = np.array(view_superpixels).astype(int) - 1

        # view_superpixels = superpixels[view_tl[1]:view_br[1],
        #                               view_tl[0]:view_br[0]]

        # plt.subplot(1, 2, 1)
        # plt.imshow(view_img)
        # plt.subplot(1, 2, 2)
        # plt.imshow(view_edge_mask)
        # plt.show()

        # debug_func.plot_view_and_label(
        #     view_img, [(x_tl, y_tl), (x_br, y_br)])

        return view_img, view_common_region, view_superpixels

    def transform_view_geom(self,
                            views,
                            labels,
                            superpixels,
                            noise_masks,
                            flip=True,
                            resize_range=None):
        """
        Implements:
            Horizontal flip
            TODO: Resize (resolution change) ?

        NOTE: Flipped views are indicated by NEGATIVE label coordinates.

        Args:
            views (list): List of view images (PIL Image)
            labels (list): List of common region rectangle coordinates in view
                           image coordiantes [ [ [x, y]_tl, [x, y]_br ], ... ].
            superpixels (list): List of superpixels maps (np.array)
            flip (bool): Perform random horizontal flip if True.
            resize_range (list): Perform random resize within range
                                 [min_size, max_size] or None.

        Returns:
            Tuple: [0] view image (PIL Image).
                   [1] common region in view coordinates [[x, y]_tl, [x, y]_br]
                   [2] superpixels (np.array).
        """
        assert len(views) == self.view_N
        assert resize_range is None or len(resize_range) == 2

        rng = default_rng()

        for idx in range(self.view_N):

            if flip:
                a = rng.uniform()
                if a > 0.5:
                    # debug_func.plot_view_and_label(views[idx], labels[idx])
                    # Flip view image
                    views[idx] = views[idx].transpose(
                        PIL.Image.FLIP_LEFT_RIGHT)
                    noise_masks[idx] = noise_masks[idx].transpose(
                        PIL.Image.FLIP_LEFT_RIGHT)
                    # Flip common region coordinates
                    w = views[idx].width
                    x_tl = labels[idx][0][0]
                    x_br = labels[idx][1][0]
                    # Flipped 'x' coord = image width - 'x' coord
                    # ==> top-left and bottom-right coordinate change position
                    labels[idx][0][0] = w - x_br
                    labels[idx][1][0] = w - x_tl
                    # NOTE: Negate coordinates to indicate flipped view
                    labels[idx] = np.negative(labels[idx]).tolist()
                    # Flip edge map
                    superpixels[idx] = np.fliplr(superpixels[idx])
                    # debug_func.plot_view_and_label(views[idx], labels[idx])
            ''' Add to separate pre-view position
            if resize_range:
                #debug_func.plot_view_and_label(views[idx], labels[idx])
                scaler = rng.uniform(resize_range[0], resize_range[1])
                # Resize image
                w = int(scaler * self.view_w)
                h = int(scaler * self.view_h)
                views[idx] = views[idx].resize((w, h))
                # Resize label
                labels[idx][0][0] *= scaler
                labels[idx][0][1] *= scaler
                labels[idx][1][0] *= scaler
                labels[idx][1][1] *= scaler
                #debug_func.plot_view_and_label(views[idx], labels[idx])
            '''

        return views, labels, superpixels, noise_masks

    @staticmethod
    def sample_view(idx, N, x, y, R, k_theta, k_r, def_size, resize_range):
        """
        Returns aboslute coordinates (x, y) of a new view samples according to
        given index.

        Sampling is done within a segment spanned by two angles corresponding
        to segment index 'idx' and radius 'R'. Center sampling bias is
        controlled by concentration parameters 'k_theta' and 'k_r'.

        NOTE: Does not check if view is within parent image.

        Args:
            idx (int): View index.
            N (int): Number of surrounding views to be generated (must be > 1).
            x (int): x coordinate of center view.
            y (int): y coordinate of center view.
            R (float): Maximum view translation limit.
            k_theta (float): Von Mises dist. concentration parameter for angle.
            k_r (float): Von Mises distr. concentration parameter for radius.
            def_size (int): Default view size (assumed to be square).
            resize_range (tuple): Min and max resize range.

        Returns:
            Absolute coordinates (x, y) of new view.
            Sampled size (int) of new view
        """
        assert N >= 2 and idx < N and idx >= 0, "Invalid view sample params"

        rng = default_rng()

        # Sample view size
        a = rng.uniform()
        zoom_coef = resize_range[1] * a + resize_range[0] * (1. - a)
        view_size = zoom_coef * def_size

        # Constrain view to enclose center of center view
        R = 0.5 * view_size

        # Angle representing one of N segments to be sampled from
        ang_range = 2. * np.pi / N
        # Segment start and end angles
        ang_0 = ang_range * idx
        # ang_1 = ang_range * (idx + 1)
        # Sample relative angle within segment and highest likelihood at center
        ang = rng.vonmises(0, k_theta)  # [-pi, pi]
        ang = ang + np.pi  # [0, 2*pi]
        ang = ang * ang_range / (2 * np.pi)  # [0, ang_range]
        # Transform to absolute angle
        ang = ang + ang_0

        # Sample radius within range [0, R] with highest likelihood for center
        r = rng.vonmises(0.5 * np.pi, k_r)  # [-pi, pi]
        r = r + np.pi  # [0, 2*pi]
        r = r * R / (2 * np.pi)  # [0, R]

        # Transform coordinates to system w. x-axis as zero angle
        ang = np.pi / 2 - ang

        view_x = r * np.cos(ang)
        view_y = r * np.sin(ang)

        view_x = int(view_x)
        view_y = int(view_y)

        # Transform relative --> absolute coordinates
        view_x += x
        view_y += y

        return view_x, view_y, view_size

    @staticmethod
    def clip_view_coordinates(x: int, y: int, H: int, W: int, view_h: int,
                              view_w: int):
        """
        Returns the coordinate pair (x, y) clipped so a view will always remain
        within parent image spanned by (H, W).

        Args:
            x (int): View center 'x' coordinate.
            y (int): View center 'y' coordinate.
            H (int): Parent image pixel height.
            W (int): Parent image pixel width.
            view_h (int): View height
            view_w (int): View width
        """
        x_min_margin = int(np.ceil(0.5 * view_w))
        y_min_margin = int(np.ceil(0.5 * view_h))
        x_max_margin = int(W - 0.5 * view_w)
        y_max_margin = int(H - 0.5 * view_h)

        if x < x_min_margin:
            x = x_min_margin
        elif x > x_max_margin:
            x = x_max_margin

        if y < y_min_margin:
            y = y_min_margin
        elif y > y_max_margin:
            y = y_max_margin

        return x, y

    def common_rectangle(self, view_centers, view_sizes):
        """
        Returns the common rectangle region spanned by all views as a list of
        (x, y) coordinates of the top-left and bottom-right corners.

        Args:
            view_centers (list): View center coordinate pairs
                                 [[x, y]_1, ... , [x, y]_N].
            view_sizes (list): View size integers [ s_1, ... , s_N ].

        Returns:
            List of top-left and bottom-right coordinate pairs spanning the
            common region rectangle [ [x, y]_tl, [x, y]_br ].
        """
        # Center view region
        half_h = int(0.5 * view_sizes[0])
        half_w = int(0.5 * view_sizes[0])

        common_x0 = view_centers[0][0] - half_w
        common_y0 = view_centers[0][1] - half_h
        common_x1 = view_centers[0][0] + half_w
        common_y1 = view_centers[0][1] + half_h

        # Iteratively reduce the common rectangle view-by-view
        N = len(view_centers)
        for idx in range(1, N):
            # Center view region
            half_h = int(0.5 * view_sizes[idx])
            half_w = int(0.5 * view_sizes[idx])

            x0 = view_centers[idx][0] - half_w
            y0 = view_centers[idx][1] - half_h
            x1 = view_centers[idx][0] + half_w
            y1 = view_centers[idx][1] + half_h

            common_x0 = max(common_x0, x0)
            common_y0 = max(common_y0, y0)

            common_x1 = min(common_x1, x1)
            common_y1 = min(common_y1, y1)

        return [[common_x0, common_y0], [common_x1, common_y1]]

    def threshold_common_region_size(self, common_rectangle):
        """
        Returns a 'common rectangle' that is thresholded to a maximum area.

        Thresholded rectangles are reduced by scaling with the center point
        being constant.

        Args:
            common_rectangle (list): List of top-left and bottom-right
                                     coordinate pairs spanning the common
                                     region rectangle in full-size image
                                     coordinates [ [x, y]_tl, [x, y]_br ].
        """
        # Calculate size
        tl, br = common_rectangle
        x_tl, y_tl = tl
        x_br, y_br = br

        dx = x_br - x_tl
        dy = y_br - y_tl
        N = dx * dy

        if N > self.max_common_N:
            # Calculate scaling coefficient for reducing common region
            scaling = np.sqrt(self.max_common_N / N)

            dx_scaled = scaling * dx
            dy_scaled = scaling * dy

            dx_half_diff = int(np.ceil((dx - dx_scaled) / 2))
            dy_half_diff = int(np.ceil((dy - dy_scaled) / 2))

            x_tl += dx_half_diff
            y_tl += dy_half_diff

            x_br -= dx_half_diff
            y_br -= dy_half_diff

            tl = [x_tl, y_tl]
            br = [x_br, y_br]
            # Replace 'common rectangle' with reduced version
            common_rectangle = [tl, br]

        return common_rectangle

    @staticmethod
    def conv_center2rect(x, y, h, w):
        """
        Converts rectangle center coordiantes (x, y) to top-left (x, y)_tl and
        bottom-right (x, y)_br coordinates.

        Args:
            x (int): Rectangle center 'x' coordinate.
            y (int):
            h (int): Rectangle pixel height.
            w (int):

        Returns:
            List of coordinates [ [x_tl, y_tl], [x_br, y_br] ].
        """
        x_tl = int(x - 0.5 * h)
        y_tl = int(y - 0.5 * w)
        x_br = int(x + 0.5 * h)
        y_br = int(y + 0.5 * w)
        return [[x_tl, y_tl], [x_br, y_br]]

    @staticmethod
    def transf_coord_img2view(x_img, y_img, view_tl_x, view_tl_y):
        """
        Args:
            x_img (int): Coordinate in image frame.
            y_img (int):
            view_tl_x (int): Top-left view coordinate in image frame.
            view_tl_y (int):

        Returns:
            Tuple of coordinates (x, y) in view coordinates.
        """
        x_view = x_img - view_tl_x
        y_view = y_img - view_tl_y
        return x_view, y_view

    def analyze_common_view_elem_count(self, img, samples_N):
        """
        Auxiliary function for analyzing expected number of 'common region'
        elements resulting from the view generation function.

        The 'common region' size depends on view size, number of generated
        views, and view generation sampling parameters.
        """
        from scipy.stats import beta

        common_region_Ns = []
        # Generate random views and compute common region size
        for _ in range(samples_N):
            _, labels, _ = self.gen_views(img)

            tl, br = labels[0]
            x_tl, y_tl = tl
            x_br, y_br = br

            common_region_N = (y_br - y_tl) * (x_br - x_tl)
            common_region_Ns.append(common_region_N)

        # Plots histogram of samples and fitted Beta distribution
        plt.hist(common_region_Ns, 40, density=True)

        a, b, loc, scale = beta.fit(common_region_Ns)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = beta.pdf(x, a, b, loc, scale)
        mean, var = beta.stats(a, b, loc, scale, moments='mv')
        plt.plot(x, p, 'k', linewidth=2)
        title = f"Mean: {mean:.2f}, Var: {var:.2f}"
        plt.title(title)

        plt.show()
