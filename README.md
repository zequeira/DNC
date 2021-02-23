# DNC: Dataset for Noise Classification

The **DNC dataset** contains 4377 environmental background noise recordings labeled according to type of noise.
The recordings are approximately equally balanced between three main categories, i.e., *"mechanic"*, *"melodic"*, and *"quiet"*. These noise categories were selected as we found in a previous study that these are the background noises that can distract users in crowdsourcing when performing tasks [[1]](#1) [[2]](#2).
The recordings were collected through the audio-web API employing different Windows and Mac computers.



| Noise Classes | Noise Category           | Files per<br>Category  |
| ------------- |:-------------:| -----:|--------:|--------:|
| coffee machine, dishwasher <br> water heater, street traffic | mechanic | 1545 |
| TV, TV-Show, music <br> radio, people    | melodic  | 1427 |
| quiet                            | quiet    | 1405 |


## Repository content

- [`audio/*.wav`](audio/)

  4377 audio recordings in WAV format (14.6.0 seconds long on average, 48.0 kHz, stereo).


## Citing

If you find this dataset useful in an academic setting please cite:
(to be updated)


## Download

Please get in touch to download the dataset: (rafael.zequeira@tu-berlin.de)


## References

<a id="1">[1]</a>
R. Zequeira Jiménez, B. Naderi, and S. Möller, "Background Environment Characteristics of Crowd-Workers from German Speaking Countries Experimental Survey on User Environment Characteristics," in 2019 Eleventh International Conference on Quality of Multimedia Experience (QoMEX), 2019, pp. 1–3. [DOI: https://doi.org/10.1109/QoMEX.2019.8743208]

<a id="2">[2]</a>
R. Zequeira Jiménez, B. Naderi, and S. Möller, "Effect of Environmental Noise in Speech Quality Assessment Studies using Crowdsourcing," in 2020 Twelfth International Conference on Quality of Multimedia Experience (QoMEX), 2020, pp. 1–6. [DOI: https://doi.org/10.1109/QoMEX48832.2020.9123144]