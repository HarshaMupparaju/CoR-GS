from pathlib import Path
import json

# Path to the directory containing the results
output_dir = Path('/mnt/2tb-hdd/Harsha/CoR-GS/output')

# dataset_name = 'llff'
# views = 2

# dataset_name = 'mipnerf360'
# views = 36

dataset_name = 'realestate'
views = 2

dataset_output_dir = output_dir / f'{dataset_name}_{views}'
scenes = dataset_output_dir.glob('*')
scenes = [scene for scene in scenes if scene.is_dir()]
scenes = [scene.name for scene in scenes]

aggregated_results = {}
for scene in scenes:
    print(f'Processing scene {scene}')
    scene_results_path = dataset_output_dir / scene / 'results.json'
    with open(scene_results_path, 'r') as f:
        scene_results = json.load(f)
        # Parse the results and average over all scenes
        if (dataset_name == 'llff' or dataset_name == 'realestate'):
            scene_results = scene_results['ours_10000']
        elif dataset_name == 'mipnerf360':
            scene_results = scene_results['ours_30000']
        for key, value in scene_results.items():
            if key not in aggregated_results:
                aggregated_results[key] = []
            aggregated_results[key].append(value)
    
# Average the results
for key, values in aggregated_results.items():
    aggregated_results[key] = sum(values) / len(values)

# Save the aggregated results
aggregated_results_path = dataset_output_dir / 'aggregated_results.json'
with open(aggregated_results_path, 'w') as f:
    json.dump(aggregated_results, f)
print(f'Aggregated results saved to {aggregated_results_path}')