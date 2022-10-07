from emb_viz import (EmbeddingVisualizerInterface, FuzzyEmbeddingVisualizer,
                     SymbolicEmbeddingVisualizer)
import pickle
import argparse
import pickle
from PIL import Image
import cv2
import os
import glob

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("img_dir", type=str)
    parser.add_argument("config_path", type=str, help="Path to config file.")
    parser.add_argument("checkpoint_path",
                        type=str,
                        help="Path to checkpoint file.")
    parser.add_argument("output_type",
                        type=str,
                        help="Output type choice (viz_emb, backbone).")
    parser.add_argument("embedding_type", type=str, help="")
    parser.add_argument("--default-config-path",
                        type=str,
                        default="vissl/config/defaults.yaml",
                        help="Path to default config file")
    parser.add_argument("--viz-thres",
                        type=float,
                        default=0.,
                        help="Masking out embeddings/symbols bellow threshold")
    args = parser.parse_args()

    img_dir = args.img_dir
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    output_type = args.output_type
    default_config_path = args.default_config_path
    viz_thres = args.viz_thres
    emb_type = args.embedding_type

    # 0. PCA mapping
    pca = pickle.load(open("pca.pkl", "rb"))

    # 1. Setup embedding visualizer module
    if emb_type == "fuzzy":
        emb_viz_module = EmbeddingVisualizerInterface(
            FuzzyEmbeddingVisualizer(
                config_path,
                default_config_path,
                checkpoint_path,
                output_type,
                True,
                pca=pca,
                viz_thres=viz_thres,
            ))
    elif emb_type == "symbolic":
        emb_viz_module = EmbeddingVisualizerInterface(
            SymbolicEmbeddingVisualizer(
                config_path,
                default_config_path,
                checkpoint_path,
                output_type,
                True,
                pca=pca,
                viz_thres=viz_thres,
            ))

    img_paths = glob.glob(os.path.join(img_dir, "*.png"))
    imgs = []
    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        imgs.append(img)

    if len(imgs) == 0:
        raise Exception(f"No images found (img_dir: {img_dir})")

    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        emb_viz = emb_viz_module.generate(img)

        #emb_viz = np.transpose(emb_viz, (2, 0, 1))
        #emb_viz = cv2.cvtColor(emb_viz, cv2.COLOR_BGR2RGB)
        emb_viz = emb_viz[..., [2, 1, 0]]
        cv2.imwrite(img_path[:-4] + "_pca.png", emb_viz)
