#!/bin/bash

cd ${HOME}/ece569-project/build

DATADIR=${HOME}/ece569-project/data

pwd
# Just running shadow removal on the main picture
./shadow_removal "$DATADIR"/plt4.ppm "$DATADIR"/Output

# Uncomment for profiling
#parallel "mpirun -n 1 ./shadow_removal" ::: `seq 0 9` 
#nvprof -f -o gfProfile.metrics --analysis-metrics ./shadow_removal "$DATADIR"/gf.ppm "$DATADIR"/Output/gf
#nvprof -f -o gfProfile.timeline ./shadow_removal "$DATADIR"/gf.ppm "$DATADIR"/Output/gf

#nvprof -f -o plant_sdProfile.metrics --analysis-metrics ./shadow_removal "$DATADIR"/plant_sd.ppm "$DATADIR"/Output/plant_sd
#nvprof -f -o plant_sdProfile.timeline ./shadow_removal "$DATADIR"/plant_sd.ppm "$DATADIR"/Output/plant_sd

#nvprof -f -o plant2Profile.metrics --analysis-metrics ./shadow_removal "$DATADIR"/plant2.ppm "$DATADIR"/Output/plant2
#nvprof -f -o plant2Profile.timeline ./shadow_removal "$DATADIR"/plant2.ppm "$DATADIR"/Output/plant2

#nvprof -f -o pltProfile.metrics --analysis-metrics ./shadow_removal "$DATADIR"/plt.ppm "$DATADIR"/Output/plt
#nvprof -f -o pltProfile.timeline ./shadow_removal "$DATADIR"/plt.ppm "$DATADIR"/Output/plt

#nvprof -f -o plt4Profile.metrics --analysis-metrics ./shadow_removal "$DATADIR"/plt4.ppm "$DATADIR"/Output/plt4
#nvprof -f -o plt4Profile.timeline ./shadow_removal "$DATADIR"/plt4.ppm "$DATADIR"/Output/plt4

#nvprof -f -o plt5Profile.metrics --analysis-metrics ./shadow_removal "$DATADIR"/plt5.ppm "$DATADIR"/Output/plt5
#nvprof -f -o plt5Profile.timeline ./shadow_removal "$DATADIR"/plt5.ppm "$DATADIR"/Output/plt5

#nvprof -f -o 1020ImageProfile.metrics --analysis-metrics ./shadow_removal "$DATADIR"/1020Image.ppm "$DATADIR"/Output/1020Image
#nvprof -f -o 1020ImageProfile.timeline ./shadow_removal "$DATADIR"/1020Image.ppm "$DATADIR"/Output/1020Image

#nvprof -f -o 4kImageProfile.metrics --analysis-metrics ./shadow_removal "$DATADIR"/4kImage.ppm "$DATADIR"/Output/4kImage
#nvprof -f -o 4kImageProfile.timeline ./shadow_removal "$DATADIR"/4kImage.ppm "$DATADIR"/Output/4kImage
