from utils.utils import load_config, load_modalities, create_gif_from_volume as brats_create_gif_from_volume
from scripts.acdc_evaluate import eval as acdc_eval
from scripts.acdc_inference import inference as acdc_inference
from scripts.acdc_inference import create_gif_from_volume as acdc_create_gif_from_volume
from scripts.acdc_inference import decode_one_hot_numpy
from scripts.brats_evaluate import eval as brats_eval
from scripts.brats_inference import inference as brats_inference
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    if args.dataset == "acdc":
        if args.type == "bsm":
            config = load_config("bsm_acdc_config.yaml")
        elif args.type == "epa":
            config = load_config("epa_acdc_config.yaml")
        if args.command == "eval":
           acdc_eval(config)
        elif args.command == "inference":
            image_pth = "data/acdc_seg/processed/val/patient136_frame01/patient136_frame01_modalities.pt"
            label_pth = "data/acdc_seg/processed/val/patient136_frame01/patient136_frame01_label.pt"
            image = load_modalities(image_pth)
            label = load_modalities(label_pth).astype(int)
            prediction = acdc_inference(config, image, cpkt=f"checkpoint/{config['model']['name']}/model.safetensors")[0]
            prediction = decode_one_hot_numpy(prediction)
            acdc_create_gif_from_volume(volume=image, prediction=label[0], gif_path="assets/acdc/volume_with_label/patient136_frame01.gif")
            acdc_create_gif_from_volume(volume=image, prediction=prediction, gif_path="assets/acdc/volume_with_prediction/patient136_frame01.gif")
    elif args.dataset == "brats":
        if args.type == "bsm":
            config = load_config("bsm_brats_config.yaml")
        elif args.type == "epa":
            config = load_config("epa_brats_config.yaml")
        if args.command == "eval":
           brats_eval(config)
        elif args.command == "inference":
            image_pth = "data/brats2021_seg/processed/val/BraTS2021_01657/BraTS2021_01657_modalities.pt"
            label_pth = "data/brats2021_seg/processed/val/BraTS2021_01657/BraTS2021_01657_label.pt"
            image = load_modalities(image_pth)
            label = load_modalities(label_pth).astype(int)
            prediction = brats_inference(config, image, cpkt=f"checkpoint/{config['model']['name']}/model.safetensors")[0]
            brats_create_gif_from_volume(volume=image, prediction=label, gif_path="assets/brats2021_seg/volume_with_label/BraTS2021_01657.gif")
            brats_create_gif_from_volume(volume=image, prediction=prediction, gif_path="assets/brats2021_seg/volume_with_prediction/BraTS2021_01657.gif")
   