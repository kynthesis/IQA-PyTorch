import argparse
import glob
import os
from pyiqa import create_metric
from tqdm import tqdm


def main():
    """Inference demo for pyiqa.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=None, help='input image/folder path.')
    parser.add_argument('-r', '--ref', type=str, default=None, help='reference image/folder path if needed.')

    args = parser.parse_args()

    metric_name = 'LPIPS'.lower()

    # set up IQA model
    iqa_model = create_metric(metric_name, metric_mode='FR')

    if os.path.isfile(args.input):
        input_paths = [args.input]
        if args.ref is not None:
            ref_paths = [args.ref]
    else:
        input_paths = sorted(glob.glob(os.path.join(args.input, '*')))
        if args.ref is not None:
            ref_paths = sorted(glob.glob(os.path.join(args.ref, '*')))


    avg_score = 0
    test_img_num = len(input_paths)
    pbar = tqdm(total=test_img_num, unit='image')
    for idx, img_path in enumerate(input_paths):
        ref_img_path = ref_paths[idx]

        score = iqa_model(img_path, ref_img_path).cpu().item()
        avg_score += score
        pbar.update(1)
    pbar.close()
    avg_score /= test_img_num
    print(f'LPIPS: {round(avg_score, 4)}')


if __name__ == '__main__':
    main()
