import argparse

from src import config
from src.DF_SLAM import DF_SLAM

def main():
    parser = argparse.ArgumentParser(
        description='Arguments for running DF_SLAM.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/DF_SLAM.yaml')
    
    df_slam = DF_SLAM(cfg, args)

    df_slam.run()

if __name__ == '__main__':
    main()
