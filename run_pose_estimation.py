import argparse
import torch
from get_rect import get_rect
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state

def main(image_path):
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load('checkpoint_iter_370000.pth', map_location='cpu')
    load_state(net, checkpoint)
    
    get_rect(net.cuda(), [image_path], 512)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run pose estimation on an image.')
    parser.add_argument('image_path', type=str, help='Path to the image file')

    args = parser.parse_args()
    main(args.image_path)
