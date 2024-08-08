import sys
sys.path.append("../")

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import PERCEPTION, PATH_TO_LABELS, PARTITION_FILES, PERCEPTION_LABELS

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--abs_threshold', type=float, default=0.0, 
                            help='Specify a threshold between [0, 1] to truncate absolute value of correlation')

    args = parser.parse_args()

    return args


def compute_atts_correlation_matrix():

    df_label = pd.read_csv(PATH_TO_LABELS[PERCEPTION])
    df_meta = pd.read_csv(PARTITION_FILES[PERCEPTION])

    df_meta.rename(columns={'Id':'subj_id'}, inplace=True)

    df = df_label.merge(df_meta, on='subj_id')
    df = df[df['Partition'].isin(['train', 'devel'])] # 'train', 'devel'

    df.drop(columns=['subj_id', 'attractive', 'charismatic', 'competitive', 'expressive', 'naive'], inplace=True)
    df.dropna(inplace=True)

    # PERCEPTION_LABELS
    atts = ['leader_like', 'confident', 'independent', 
            'assertiv', 'dominant', 'aggressive', 
            'risk',  
            'arrogant', 
            'enthusiastic', 
            'friendly', 'likeable', 'sincere', 'collaborative', 'kind', 'warm', 'good_natured', 
            ]

    title_atts = [att.replace('_', ' ').title() for att in atts]


    df = df[atts]
    df.columns = [title_atts]

    correlation_matrix = df.corr()

    truncated_corr = correlation_matrix.map(lambda x: 0 if abs(x) < args.abs_threshold else x)

    return truncated_corr


def main(args):
    
    corr = compute_atts_correlation_matrix()

    plt.figure(figsize=(10, 8))

    ax = sns.heatmap(corr, annot=False, cmap='gray_r', vmin=-1, vmax=1, center=0)

    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.xticks(fontsize=13, rotation=45)
    plt.yticks(fontsize=13)

    labels = ax.get_xticklabels()
    for label in labels:
        label.set_ha('right')
        label.set_position((label.get_position()[0] - 0.5, label.get_position()[1]))

    rects = [
        (0, 0, 3, 3),
        (3, 3, 3, 3),
        (9, 9, 7, 7),
    ] 

    for (x, y, width, height) in rects:
        ax.add_patch(patches.Rectangle((x, y), width, height, linewidth=3, edgecolor='yellow', facecolor='none'))

    plt.savefig('heatmap.png', format='png', bbox_inches='tight')


if __name__ == '__main__':
    
    args = parse_args()

    main(args)