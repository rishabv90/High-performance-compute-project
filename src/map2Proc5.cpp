#ifndef SMALL_CALC
#define SMALL_CALC

/*
	float shadowavg_red = deviceShadowRedArray[0] / deviceErodedShadow[0];
	float shadowavg_green = deviceShadowGreenArray[0] / deviceErodedShadow[0];
	float shadowavg_blue = deviceShadowBlueArray[0] / deviceErodedShadow[0];

	float litavg_red = deviceLightRedArray[0] / deviceErodedLight[0];
	float litavg_green = deviceLightGreenArray[0] / deviceErodedLight[0];
	float litavg_blue = deviceLightBlueArray[0] / deviceErodedLight[0];
	
	float ratio_red = litavg_red / shadowavg_red - 1;
	float ratio_green = litavg_green / shadowavg_green - 1;
	float ratio_blue = litavg_blue / shadowavg_blue - 1;
	
	
	litavg_blue            = 0.6213
      litavg_green          = 0.5891
      litavg_red               = 0.5932
      shadowavg_blue     =  0.3427
      shadowavg_green    =  0.2690
      shadowavg_red         =  0.2799
*/
__global__ void smallCalc/*V2*/(float* deviceShadowRedArray, float* deviceShadowGreenArray, float* deviceShadowBlueArray, float* deviceLightRedArray, 
float* deviceLightGreenArray, float* deviceLightBlueArray, float* deviceErodedLight, float* deviceErodedShadow) {//calculates ratios from sums and stops the need to transfer data back to cpu, and uses a 3 blocks for 3 calculations
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	if(x == 0){
		float shadowavg_red = deviceShadowRedArray[0] / deviceErodedShadow[0];
		float litavg_red = deviceLightRedArray[0] / deviceErodedLight[0];

		deviceShadowRedArray[0] = litavg_red / shadowavg_red - 1; //ratio red

	}
	else if(x == 1){
		float shadowavg_green = deviceShadowGreenArray[0] / deviceErodedShadow[0];

		float litavg_green = deviceLightGreenArray[0] / deviceErodedLight[0];
		deviceShadowGreenArray[0] = litavg_green / shadowavg_green - 1; //ratio green
	}
	else{
		float shadowavg_blue = deviceShadowBlueArray[0] / deviceErodedShadow[0];

		float litavg_blue = deviceLightBlueArray[0] / deviceErodedLight[0];
		deviceShadowBlueArray[0] = litavg_blue / shadowavg_blue - 1; //ratio blue
	}
}





__global__ void smallCalcV1(float* deviceShadowRedArray, float* deviceShadowGreenArray, float* deviceShadowBlueArray, float* deviceLightRedArray, 
float* deviceLightGreenArray, float* deviceLightBlueArray, float* deviceErodedLight, float* deviceErodedShadow) {//calculates ratios from sums and stops the need to transfer data back to cpu, and prints values for debugging other processes
	
		
}

#endif
