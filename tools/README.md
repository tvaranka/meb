# Tools

[create_sym_links.sh](create_sym_links.sh) is a script that can be used to create symbolic links from your data to the dataset folder. Before running it needs to be modified according to your spefications.
```sh
tools/create_sym_links.sh
```

[dataset_setup_test.py](dataset_setup_test.py) provides a way to test whether your datasets have been correctly setup. The script works with command line arguments.
```sh
python tools/dataset_setup_test.py --dataset_name Casme --data_type original
```

[extract_casme3_part_a.py](extract_casme3_part_a.py) extracts the micro-expression data from the entire part_A. As the provided data for CAS(ME)^3 contains the entire image sequences, only the ME parts are extracted.
```sh
python tools/extract_casme3_part_a.py --excel "data/CASME3/cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx" --casme3_original "data/casme3/part_a/data/part_A_split/part_A/" --casme3_me "data/casme3/ME_A/"
```
[extract_optical_flow.py](extract_optical_flow.py) extracts optical flow from a given dataset.
```sh
python tools/extract_optical_flow.py --dataset_name Casme --save_location data/Casme
```
Further instructions for extracting the optical flow. [Secrets of optical flow](https://cs.brown.edu/people/dqsun/pubs/cvpr_2010_flow.pdf) is used due to its accuracy and the code can be downloaded from [here](http://files.is.tue.mpg.de/black/src/ijcv_flow_code.zip). The zip file needs to be extracted to the *tools* folder and matlab engine needs to be installed using `pip install matlabengine`. See [Matlab engine](https://se.mathworks.com/help/matlab/matlab-engine-for-python.html) for issues.
