"""
This code generates 14 MFCC *.cvs datasets with MFCC coefficients varying from 5 to 31 (odd numbers).
The audio files are read from "audio_files" and each *.csv dataset is stored in "mfcc_data".

As the number of audio files is large (4377), as well as each of the resulting *.csv file, then a
High-Performance-Computing-Cluster (HPC-Cluster) was used, employing parallelization.
The file "hpc_job.sh" contains the code for executing this file in an HPC-Cluster.
"""
import numpy as np
import pandas as pd
from itertools import repeat
import sklearn.preprocessing as pp
from multiprocessing import Pool
import multiprocessing as mp
from glob import glob
import librosa


def calculate_mfcc(file, i):
    print(str(mp.current_process())+' and '+file)

    y, sr = librosa.load(file, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=i, dct_type=2, norm='ortho')

    mfcc = np.array(mfcc)
    mfcc = mfcc.transpose()

    # Normalization
    scaler = pp.StandardScaler()
    mfcc_Norm = scaler.fit_transform(mfcc)

    str_file = file.split('/')[-1]
    # if executed on Windows
    # str_file = file.split('\\')[-1]
    label = np.full((np.shape(mfcc_Norm)[0], 1), str_file.split(' _ ')[0])
    label_level = np.full((np.shape(mfcc_Norm)[0], 1), str_file.split(' _ ')[3])

    if label[0] == 'quiet':
        label_g = 'Quiet'
    elif label[0] == 'TV' or label[0] == 'Music' or label[0] == 'Radio':
        label_g = 'Melodic'
    else:
        label_g = 'Mechanic'
    label_group = np.full((np.shape(mfcc_Norm)[0], 1), label_g)

    student = np.full((np.shape(mfcc_Norm)[0], 1), str_file.split(' _ ')[1])
    file_name = np.full((np.shape(mfcc_Norm)[0], 1), str_file.split('.')[0])
    mfcc_Norm = np.hstack((mfcc_Norm, label, label_group, label_level, file_name, student))
    return mfcc_Norm


if __name__ == '__main__':
    audio_path = '../audio_files/*/'
    mfcc_data = '../mfcc_data/'
    print("Number of processors: ", mp.cpu_count())

    # i controls the number of computed MFCC coefficients
    for i in range(5, 31, 1):
        # Get the number of available cores
        P = Pool(mp.cpu_count())

        # Each core gets a file and the function "calculate_mfcc" to calculate the MFCC features
        coefficients = P.starmap(calculate_mfcc, zip(glob(audio_path+'/*.wav'), repeat(i)))

        features = np.vstack(coefficients)

        P.close()

        headers = ['MFCC{:01d}'.format(j) for j in range(1, i+1, 1)]
        headers = headers + ['LABEL', 'LABEL_GROUP', 'LABEL_LEVEL', 'FILE', 'STUDENT']

        featuresDF = pd.DataFrame(data=features, columns=headers)

        featuresDF.to_csv(mfcc_data+'featuresNormalized_MFCC_{:01d}.csv'.format(i), sep=';', float_format='%.3f')