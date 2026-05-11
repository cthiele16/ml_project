import argparse
import eda
import preprocessing
from phases import phase1, phase2, phase3, phase4, phase5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["eda", "preprocessing", "phase1", "phase2", "phase3", "phase4", "phase5"])
    args = parser.parse_args()

    if args.phase == "eda":
        eda.run()
    elif args.phase == "preprocessing":
        preprocessing.run()
    elif args.phase == "phase1":
        phase1.run()
    elif args.phase == "phase2":
        phase2.run()
    elif args.phase == "phase3":
        phase3.run()
    elif args.phase == "phase4":
        phase4.run()
    elif args.phase == "phase5":
        phase5.run()
