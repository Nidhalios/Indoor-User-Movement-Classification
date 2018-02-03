"""
Project: Indoor-User-Movement-Classification
Author: Nidhalios
Created On: 2/3/18
"""

import pandas as pd

NB_OBS = 314


def frange(a, b, step):
    """
    A generator function for float ranges
    """
    while a < b:
        yield a
        a += step


def load_data(path='../data'):
    """
    Loading and joining the RSS sequences, targets, dataset groups and the paths data.
    All of the consolidated data will be stored in a list of dictionaries 'series'
    """

    series = []

    groupData = pd.read_csv(path + '/groups/MovementAAL_DatasetGroup.csv', sep=',')
    groupData.sort_values(by='sequence_ID', ascending=True, inplace=True)
    groupData.index = groupData['sequence_ID']
    # print(groupData.head())
    # print(groupData.info())
    # print("Missing Group Data:",groupData.isnull().values.any())

    pathData = pd.read_csv(path + '/groups/MovementAAL_Paths.csv', sep=',')
    pathData.sort_values(by='sequence_ID', ascending=True, inplace=True)
    pathData = pathData.drop(pathData.columns[[2, 3]], axis=1)
    pathData.index = pathData['sequence_ID']
    # print(pathData.head())
    # print(pathData.info())
    # print("Missing Path Data:",pathData.isnull().values.any())

    targetData = pd.read_csv(path + '/dataset/MovementAAL_target.csv', sep=',')
    targetData.sort_values(by='sequence_ID', ascending=True, inplace=True)
    targetData.index = targetData['sequence_ID']
    # print(targetData.head())
    # print(targetData.info())
    # print("Missing Target Data:",targetData.isnull().values.any())

    # Iterate through the 314 sequences CSV files to extract the series,
    # associate them with the group_id and the corresponding path_id.
    for i in range(1, NB_OBS + 1):
        df = pd.read_csv(path + '/dataset/MovementAAL_RSS_{}.csv'.format(i), sep=',', header=0,
                         names=['anchor1', 'anchor2', 'anchor3', 'anchor4'])
        rng = list(frange(0, df.shape[0] * 0.125, 0.125))
        df.index = rng
        # print("i : ",i," Missing : ",df.isnull().values.any())
        if not df.isnull().values.any():
            series.append([i,
                           df,
                           groupData.loc[[i]].values.tolist()[0][1],
                           pathData.loc[[i]].values.tolist()[0][1],
                           targetData.loc[[i]].values.tolist()[0][1]
                           ])

    names = ['id', 'series', 'group_id', 'path_id', 'target']
    return pd.DataFrame(data=series, columns=names)


def main():
    temp = load_data()
    print(temp.head(1).values)


if __name__ == '__main__': main()
