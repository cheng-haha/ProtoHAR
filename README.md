<p align=center> <img src="cover.png" width = 60%/> </p>


<p align="center"><em>ProtoHAR: Prototype Guided Personalized Federated Learning for Human Activity Recognition</em></p>

<p align="center"><a href="https://github.com/cheng-haha">Dongzhou Cheng</a></p>


## Abstract
TODO. We hope this research can bring a new perspective to the HAR community.

## The code will be coming soon

## Benchmark
We offer a benchmark for USC-HAD and HARBOX. 

## Usage
1. git clone the repo
```
git clone https://github.com/cheng-haha/ProtoHAR.git
```

2. Enter the current folder
```
cd yourfolder
```
3. Generate heterogeneous data sets

```
python  data/uschad/uschad_subdata.py
python  data/harbox/harbox_subdata.py
```
4. run
````
bash runexp.sh
````

## Run details 
1. USC-HAD has 14 clients, HARBOX has 120 clients
2. the learning rate of USC is 0.001, HARBOX:0.01
