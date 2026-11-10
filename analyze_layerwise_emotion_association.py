#!/usr/bin/env python3
"""Create layer-wise single/shared-emotion reports from selected neuron IDs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def build_rows(data: dict) -> tuple[list[dict], list[float], list[np.ndarray]]:
    percentages = [float(value) for value in data['percentages']]
    layers, width, emotions = int(data['num_layers']), int(data['neurons_per_layer']), len(data['labels'])
    rows, matrices = [], []
    for percentage in percentages:
        key = f'{percentage:g}%'
        matrix = np.zeros((layers, emotions + 1), dtype=np.int64)
        for layer_index in range(layers):
            by_emotion = data['selected_neurons'][key][f'layer_{layer_index}']
            associations: dict[int, set[str]] = {}
            for emotion, neuron_ids in by_emotion.items():
                for neuron in neuron_ids:
                    associations.setdefault(int(neuron), set()).add(emotion)
            matrix[layer_index, 0] = width - len(associations)
            for associated_emotions in associations.values():
                matrix[layer_index, len(associated_emotions)] += 1
            for degree, count in enumerate(matrix[layer_index]):
                rows.append({
                    'percentage': percentage,
                    'layer': layer_index,
                    'association_degree': degree,
                    'association_label': 'not_selected' if degree == 0 else
                                         'single_emotion' if degree == 1 else f'shared_{degree}_emotions',
                    'neuron_count': int(count),
                    'percentage_of_layer_neurons': float(count / width * 100.0),
                })
        matrices.append(matrix[:, 1:])  # PNG shows only selected, degree 1..E.
    return rows, percentages, matrices


def write_png(percentages: list[float], matrices: list[np.ndarray], path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(percentages), figsize=(4 * len(percentages), 9), squeeze=False)
    maximum = max(int(matrix.max()) for matrix in matrices) or 1
    image = None
    for axis, percentage, matrix in zip(axes[0], percentages, matrices):
        image = axis.imshow(matrix, aspect='auto', cmap='viridis', vmin=0, vmax=maximum)
        axis.set_title(f'Top {percentage:g}%')
        axis.set_xlabel('Associated emotions')
        axis.set_xticks(range(matrix.shape[1]), [str(degree) for degree in range(1, matrix.shape[1] + 1)])
        axis.set_yticks(range(matrix.shape[0]), [str(layer) for layer in range(matrix.shape[0])])
        if axis is axes[0, 0]:
            axis.set_ylabel('Transformer layer')
    figure.colorbar(image, ax=axes.ravel().tolist(), label='Neuron count')
    figure.suptitle('Layer-wise emotion-association degree of selected Sarvam neurons', y=0.98)
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-file', default='results/llama31_8b_neuron_analysis/top_selectivity_neurons.json')
    parser.add_argument('--output-csv', default='results/llama31_8b_neuron_analysis/layerwise_emotion_association_report.csv')
    parser.add_argument('--output-png', default='results/llama31_8b_neuron_analysis/layerwise_emotion_association_report.png')
    args = parser.parse_args()
    data = json.loads(Path(args.input_file).read_text(encoding='utf-8'))
    required = {'percentages', 'num_layers', 'neurons_per_layer', 'labels', 'selected_neurons'}
    if missing := required - data.keys():
        raise ValueError(f'Input JSON is missing keys: {sorted(missing)}')
    rows, percentages, matrices = build_rows(data)
    csv_path, png_path = Path(args.output_csv), Path(args.output_png)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    write_png(percentages, matrices, png_path)
    print(f'Wrote {csv_path}')
    print(f'Wrote {png_path}')


if __name__ == '__main__':
    main()
