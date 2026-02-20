import os
import argparse
from glob import glob

import cv2
import torch
from tqdm import tqdm

from config import Config
from dataset import MyData
from evaluation.metrics import evaluator
from models.adasnet import AdasNet
from utils import save_tensor_img, check_state_dict


config = Config()


def render_table(headers, rows):
    rows_str = [[str(c) for c in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in rows_str:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def build_row(items):
        return '| ' + ' | '.join(item.ljust(widths[idx]) for idx, item in enumerate(items)) + ' |'

    sep = '+-' + '-+-'.join('-' * w for w in widths) + '-+'
    lines = [sep, build_row(headers), sep]
    for row in rows_str:
        lines.append(build_row(row))
    lines.append(sep)
    return '\n'.join(lines)


def run_inference(model, data_loader_test, pred_root, method, testset, device):
    model_training = model.training
    if model_training:
        model.eval()

    model.half()
    iterator = tqdm(data_loader_test, total=len(data_loader_test)) if config.verbose_eval else data_loader_test
    for batch in iterator:
        inputs = batch[0].to(device).half()
        label_paths = batch[-1]

        with torch.no_grad():
            scaled_preds = model(inputs)[-1].sigmoid()

        out_dir = os.path.join(pred_root, method, testset)
        os.makedirs(out_dir, exist_ok=True)

        for idx_sample in range(scaled_preds.shape[0]):
            res = torch.nn.functional.interpolate(
                scaled_preds[idx_sample].unsqueeze(0),
                size=cv2.imread(label_paths[idx_sample], cv2.IMREAD_GRAYSCALE).shape[:2],
                mode='bilinear',
                align_corners=True,
            )
            save_tensor_img(
                res,
                os.path.join(out_dir, label_paths[idx_sample].replace('\\', '/').split('/')[-1]),
            )

    if model_training:
        model.train()


def build_model():
    if str(config.model).lower() == 'adasnet':
        return AdasNet(bb_pretrained=False)
    raise ValueError(f'Unsupported model type: {config.model}')


def get_weights_list(ckpt_folder, ckpt):
    weights_lst = sorted(
        glob(os.path.join(ckpt_folder, '*.pth')) if ckpt_folder else [ckpt],
        key=lambda x: int(x.split('epoch_')[-1].split('.pth')[0]),
        reverse=True,
    )
    return weights_lst


def run_inference_stage(args):
    device = config.device
    if args.ckpt_folder:
        print(f'Testing with models in {args.ckpt_folder}')
    else:
        print(f'Testing with model {args.ckpt}')

    model = build_model()
    weights_lst = get_weights_list(args.ckpt_folder, args.ckpt)

    for testset in args.testsets.split('+'):
        print(f'>>>> Testset: {testset}...')
        data_loader_test = torch.utils.data.DataLoader(
            dataset=MyData(testset, data_size=config.size, is_train=False),
            batch_size=config.batch_size_valid,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        for weights in weights_lst:
            epoch = int(weights.strip('.pth').split('epoch_')[-1])
            if epoch % args.infer_epoch_step != 0:
                continue

            print(f'\tInferencing {weights}...')
            state_dict = torch.load(weights, map_location='cpu', weights_only=True)
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
            model = model.to(device)

            run_inference(
                model=model,
                data_loader_test=data_loader_test,
                pred_root=args.pred_root,
                method='--'.join([w.rstrip('.pth') for w in weights.split(os.sep)[-2:]]),
                testset=testset,
                device=device,
            )


def get_model_list(pred_root, model_lst_text):
    if model_lst_text:
        return [m.strip() for m in model_lst_text.split('+') if m.strip()]

    try:
        return [
            m for m in sorted(
                os.listdir(pred_root),
                key=lambda x: int(x.split('epoch_')[-1]),
                reverse=True,
            )
        ]
    except Exception:
        return sorted(os.listdir(pred_root))


def check_integrity(args, model_lst):
    for data_name in args.data_lst.split('+'):
        for model_name in model_lst:
            gt_pth = os.path.join(args.gt_root, data_name)
            pred_pth = os.path.join(args.pred_root, model_name, data_name)
            if not (os.path.isdir(gt_pth) and os.path.isdir(pred_pth)):
                print(f'Skip integrity check: missing {gt_pth} or {pred_pth}')
                continue
            if sorted(os.listdir(gt_pth)) != sorted(os.listdir(pred_pth)):
                print(len(sorted(os.listdir(gt_pth))), len(sorted(os.listdir(pred_pth))))
                print(f'The {data_name} dataset of {model_name} model is not matching to the ground-truth')


def run_eval_stage(args):
    model_lst = get_model_list(args.pred_root, args.model_lst)

    if args.check_integrity:
        check_integrity(args, model_lst)
    else:
        print('>>> skip check the integrity of each candidates')

    for data_name in args.data_lst.split('+'):
        pred_data_dir = sorted(glob(os.path.join(args.pred_root, model_lst[0], data_name))) if model_lst else []
        if not pred_data_dir:
            print(f'Skip dataset {data_name}.')
            continue

        gt_src = os.path.join(args.gt_root, data_name)
        gt_paths = sorted(glob(os.path.join(gt_src, 'gt', '*')))

        print('#' * 20, data_name, '#' * 20)
        filename = os.path.join(args.save_dir, f'{data_name}_eval.txt')

        headers = [
            'Dataset', 'Method', 'Smeasure', 'wFmeasure', 'meanFm', 'meanEm',
            'maxEm', 'MAE', 'maxFm', 'adpEm', 'adpFm', 'HCE', 'mBA', 'maxBIoU', 'meanBIoU',
            'mAP', 'Dice', 'IoU', 'Precision', 'Recall', 'Specificity', 'Accuracy'
        ]
        rows = []

        for model_name in model_lst:
            print(f'\tEvaluating model: {model_name}...')
            pred_paths = [
                p.replace(args.gt_root, os.path.join(args.pred_root, model_name)).replace('/gt/', '/')
                for p in gt_paths
            ]
            em, sm, fm, mae, mse, wfm, hce, mba, biou, common = evaluator(
                gt_paths=gt_paths,
                pred_paths=pred_paths,
                metrics=args.metrics.split('+'),
                verbose=config.verbose_eval,
            )

            scores = [
                sm.round(3),
                wfm.round(3),
                fm['curve'].mean().round(3),
                em['curve'].mean().round(3),
                em['curve'].max().round(3),
                mae.round(3),
                fm['curve'].max().round(3),
                em['adp'].round(3),
                fm['adp'].round(3),
                int(hce.round()),
                mba.round(3),
                biou['curve'].max().round(3),
                biou['curve'].mean().round(3),
                common['map'].round(3),
                common['dice']['mean'].round(3),
                common['iou']['mean'].round(3),
                common['precision']['mean'].round(3),
                common['recall']['mean'].round(3),
                common['specificity']['mean'].round(3),
                common['accuracy']['mean'].round(3),
            ]

            for idx_score, score in enumerate(scores):
                scores[idx_score] = '.' + format(score, '.3f').split('.')[-1] if score <= 1 else format(score, '<4')

            records = [data_name, model_name] + scores
            rows.append(records)

            table_text = render_table(headers, rows)

            with open(filename, 'w+', encoding='utf-8') as file_to_write:
                file_to_write.write(table_text + '\n')

        if rows:
            print(table_text)
        else:
            print(render_table(headers, rows))


def resolve_default_ckpt_folder():
    folders = sorted(glob(os.path.join('ckpt', '*')))
    return folders[-1] if folders else None


def parse_args():
    parser = argparse.ArgumentParser(description='Inference + Evaluation unified script')
    parser.add_argument('--mode', choices=['infer', 'eval', 'all'], default='all', help='Run inference/eval/all')

    parser.add_argument('--ckpt', type=str, default=None, help='Single model checkpoint path')
    parser.add_argument('--ckpt_folder', type=str, default=resolve_default_ckpt_folder(), help='Checkpoint folder path')
    parser.add_argument('--pred_root', type=str, default='e_preds', help='Prediction output root')
    parser.add_argument('--testsets', type=str, default=config.testsets.replace(',', '+'), help='Inference testsets')
    parser.add_argument('--infer_epoch_step', type=int, default=1, help='Infer every N epochs')

    parser.add_argument('--gt_root', type=str, default=os.path.join(config.data_root_dir, config.task), help='Ground-truth root')
    parser.add_argument('--data_lst', type=str, default=config.testsets.replace(',', '+'), help='Eval datasets joined by +')
    parser.add_argument('--save_dir', type=str, default='e_results', help='Evaluation results output dir')
    parser.add_argument('--check_integrity', action='store_true', help='Check GT and prediction file integrity')
    parser.add_argument('--metrics', type=str, default='+'.join(['S', 'MAE', 'E', 'F', 'WF', 'MBA', 'BIoU', 'HCE', 'mAP', 'DICE', 'IOU', 'PREC', 'REC', 'SPEC', 'ACC']), help='Metrics joined by +')
    parser.add_argument('--model_lst', type=str, default=None, help='Model names in pred_root joined by + for eval')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')

    os.makedirs(args.pred_root, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode in ('infer', 'all'):
        if not args.ckpt_folder and not args.ckpt:
            raise ValueError('Please provide --ckpt_folder or --ckpt for inference mode.')
        run_inference_stage(args)

    if args.mode in ('eval', 'all'):
        run_eval_stage(args)


if __name__ == '__main__':
    main()
