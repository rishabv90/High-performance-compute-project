#ifndef MAP1PROC5
#define MAP1PROC5

#define CHANNELS 3 // we have 3 channels corresponding to RGB
__global__ void map1Proc5(float * redData, float * greenData, float * blueData, float * eroded_shadow, float * erroded_light, float * shadowRedArray, float * shadowGreenArray, float * shadowBlueArray, float * lightRedArray, float * lightGreenArray, float * lightBlueArray, int width, int height) {//this kernel does the first map in proc 5 multiplication, this kernel reduces register trafic
	
}


__global__ void map1Proc5V1(float * inputImage, float * eroded_shadow, float * erroded_light, float * shadowRedArray, float * shadowGreenArray, float * shadowBlueArray, float * lightRedArray, float * lightGreenArray, float * lightBlueArray, int width, int height) {//this kernel does the first map in proc 5 multiplication
	
}

#endif
