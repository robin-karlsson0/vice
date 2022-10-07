#echo
#echo exp138_c256
#echo
#python tools/test.py configs/vice_cluster/cluster_coco-stuff164k_vissl_exp138_c256_highres_faiss.py /home/robin/projects/vissl/experiments/low_res/sc_exp175/model_final_checkpoint_phase11.torch --eval mIoU
echo
echo exp138_c128
echo
python tools/test.py configs/vice_cluster/cluster_coco-stuff164k_vissl_exp138_c128_highres_faiss.py /home/robin/projects/vissl/experiments/low_res/sc_exp175/model_final_checkpoint_phase11.torch --eval mIoU
echo
echo exp138_c45
echo
python tools/test.py configs/vice_cluster/cluster_coco-stuff164k_vissl_exp138_c45_highres_faiss.py /home/robin/projects/vissl/experiments/low_res/sc_exp175/model_final_checkpoint_phase11.torch --eval mIoU
echo
echo exp138_c27
echo
python tools/test.py configs/vice_cluster/cluster_coco-stuff164k_vissl_exp138_c27_highres_faiss.py /home/robin/projects/vissl/experiments/low_res/sc_exp175/model_final_checkpoint_phase11.torch --eval mIoU

echo
echo exp147_c256
echo
python tools/test.py configs/vice_cluster/cluster_coco-stuff164k_vissl_exp147_c256_lowress_faiss.py /home/robin/projects/vissl/experiments/low_res/sc_exp175/model_final_checkpoint_phase11.torch --eval mIoU
echo
echo exp147_c128
echo
python tools/test.py configs/vice_cluster/cluster_coco-stuff164k_vissl_exp147_c128_lowress_faiss.py /home/robin/projects/vissl/experiments/low_res/sc_exp175/model_final_checkpoint_phase11.torch --eval mIoU
echo
echo exp147_c45
echo
python tools/test.py configs/vice_cluster/cluster_coco-stuff164k_vissl_exp147_c45_lowress_faiss.py /home/robin/projects/vissl/experiments/low_res/sc_exp175/model_final_checkpoint_phase11.torch --eval mIoU
echo
echo exp147_c27
echo
python tools/test.py configs/vice_cluster/cluster_coco-stuff164k_vissl_exp147_c27_lowress_faiss.py /home/robin/projects/vissl/experiments/low_res/sc_exp175/model_final_checkpoint_phase11.torch --eval mIoU
