# ProtoHAR
This is a PyTorch implementation of the paper 

## The code will be coming soon

## Benchmark
We offer a benchmark for USC-HAD and HARBOX. 


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
4. Usage
````
bash runexp.sh
````

## Run details 
1. USC-HAD has 14 clients, HARBOX has 120 clients
2. the learning rate of USC is 0.001, HARBOX:0.01
