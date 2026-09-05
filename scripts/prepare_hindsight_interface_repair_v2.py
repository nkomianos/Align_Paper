"""One-time source creation for the prospective common-instruction repair."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFIX = ('This is a hypothetical preference-identification task. Your answer identifies the user\'s preferred listed option. '
          'Treat all listed product descriptions as valid hypothetical alternatives, regardless of technical compatibility or real-world availability. '
          'Use supplied preference evidence when present. Reply with the option\'s displayed letter only.\n\n')


def main():
    old_cfg = ROOT / 'configs/hindsight_pahf_reduced_dev_v1.json'
    cfg = json.loads(old_cfg.read_text())
    cfg.update(version='endo-pahf-reduced-dev-v2-common-task-instruction-20260905',
               common_task_prefix=PREFIX, interface_repair_number=1)
    outputs = {ROOT / 'configs/hindsight_pahf_reduced_dev_v2.json': json.dumps(cfg, indent=2) + '\n'}
    for stem in ('run', 'verify'):
        original = (ROOT / f'scripts/{stem}_hindsight_pahf_reduced_dev.py').read_text()
        revised = original.replace('configs/hindsight_pahf_reduced_dev_v1.json', 'configs/hindsight_pahf_reduced_dev_v2.json')
        revised = revised.replace('verify_hindsight_pahf_reduced_dev.py', 'verify_hindsight_pahf_reduced_dev_v2.py')
        if stem == 'run':
            needle = '{"role": "user", "content": str(text)}'
            assert revised.count(needle) == 1
            revised = revised.replace(needle, '{"role": "user", "content": cfg["common_task_prefix"] + str(text)}')
        else:
            needle = '        tokenization_receipt=verify_tokenization_backend('
            assert revised.count(needle) == 1
            revised = revised.replace(needle, '        text_by_arm={arm:{id:cfg["common_task_prefix"]+text for id,text in rows.items()} for arm,rows in text_by_arm.items()}\n' + needle)
        outputs[ROOT / f'scripts/{stem}_hindsight_pahf_reduced_dev_v2.py'] = revised
    launch = (ROOT / 'scripts/launch_hindsight_reduced_remote.py').read_text()
    launch = launch.replace('scripts/run_hindsight_pahf_reduced_dev.py', 'scripts/run_hindsight_pahf_reduced_dev_v2.py')
    launch = launch.replace("'runs/reduced_dev_v1'", "'runs/reduced_dev_v2'")
    launch = launch.replace('logs/reduced_dev_v1', 'logs/reduced_dev_v2')
    outputs[ROOT / 'scripts/launch_hindsight_reduced_remote_v2.py'] = launch
    if any(p.exists() for p in outputs):
        raise FileExistsError('repair source already created; do not overwrite')
    for path, content in outputs.items():
        path.write_text(content, encoding='utf-8', newline='\n')
    print('Created v2 source only; no model/data execution.')


if __name__ == '__main__':
    main()
