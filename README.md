# DNC: Dataset for Noise Classification

The **DNC dataset** contains 4377 environmental background noise recordings labeled according to type of noise.
The recordings are approximately equally balanced between three main categories, i.e., *"mechanic"*, *"melodic"*, and *"quiet"*. These noise categories were selected as we found in a previous study that these are the background noises that can distract users in crowdsourcing when performing tasks [[1]](#1) [[2]](#2).
The recordings were collected through the audio-web API employing different Windows and Mac computers.



| Noise Classes | Noise Category| Files per<br>Category  |
| ------------- |:-------------:| ----------------------:|
| coffee machine, dishwasher <br> water heater, street traffic | mechanic | 1545 |
| TV, TV-Show, music <br> radio, people    | melodic  | 1427 |
| quiet                            | quiet    | 1405 |


## Repository content

- [`audio/*.wav`](audio/)

  4377 audio recordings in WAV format (14.6.0 seconds long on average, 48.0 kHz, stereo) with the following naming convention:
  
  `{NOISE_TYPE} _ {TIMESTAMP/ID}.wav`
  
  - `{NOISE_TYPE}` - type of environmental background noise,
  
- [FeatureGenerator](FeatureGenerator/)

  Code to generate 14 MFCC \*.cvs datasets with MFCC coefficients varying from 5 to 31 (odd numbers).

- [NoiseClassification](NoiseClassification/)

  This folder contains multiple \*.py files each containing different classifiers testing their accuracy on environmental noise classification.
  Classifiers implementation from the "scikit-learn" toolkit.

  The classifiers under test were distributed in different \*.py files so that each could be submitted as a separate job to the HPC-Cluster. 
  I proceeded in this way because it was a computationally expensive task, as each classifier was tested 14 times (on each \*.csv dataset).

## Citing

If you find this dataset useful in an academic setting please cite:
(to be updated)


## Download

The dataset can be downloaded as a single .zip file (~3.92 GB):

**[Download DNC dataset](https://depositonce.tu-berlin.de/bitstream/11303/12788/2/audios.zip)**


## References

<a id="1">[1]</a>
R. Zequeira Jiménez, B. Naderi, and S. Möller, "Background Environment Characteristics of Crowd-Workers from German Speaking Countries Experimental Survey on User Environment Characteristics," in 2019 Eleventh International Conference on Quality of Multimedia Experience (QoMEX), 2019, pp. 1–3. [DOI: https://doi.org/10.1109/QoMEX.2019.8743208]

<a id="2">[2]</a>
R. Zequeira Jiménez, B. Naderi, and S. Möller, "Effect of Environmental Noise in Speech Quality Assessment Studies using Crowdsourcing," in 2020 Twelfth International Conference on Quality of Multimedia Experience (QoMEX), 2020, pp. 1–6. [DOI: https://doi.org/10.1109/QoMEX48832.2020.9123144]