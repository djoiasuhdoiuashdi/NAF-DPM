from config import load_config
import argparse
from Binarization.src.trainer import Trainer
from Binarization.src.tester import Tester

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./Binarization/test.yml', help='path to the test.yaml file')
    args = parser.parse_args()
    config = load_config(args.config)
    print('Config loaded')
    mode = config.MODE

    if mode == 0:
        print("--------------------------")
        print('Start Testing')
        print("--------------------------")

        tester = Tester(config)
        tester.test()

        print("--------------------------")
        print('Testing complete')
        print("--------------------------")

    elif mode == 1:
        print("--------------------------")
        print('Start Training')
        print("--------------------------")

        trainer = Trainer(config)
        trainer.train()

        print("--------------------------")
        print('Training complete')
        print("--------------------------")

if __name__ == "__main__":
    main()