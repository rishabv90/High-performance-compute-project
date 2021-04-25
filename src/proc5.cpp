#ifndef PROC5
#define PROC5

  /*
  %% Finding average channel values in shadow/light areas for every channel
  shadowavg_red = sum(sum(image_double(:,:,1).*eroded_gray_shadow_mask)) / sum(sum(eroded_gray_shadow_mask));
  shadowavg_green = sum(sum(image_double(:,:,2).*eroded_gray_shadow_mask)) / sum(sum(eroded_gray_shadow_mask));
  shadowavg_blue = sum(sum(image_double(:,:,3).*eroded_gray_shadow_mask)) / sum(sum(eroded_gray_shadow_mask));

  litavg_red = sum(sum(image_double(:,:,1).*eroded_gray_light_mask)) / sum(sum(eroded_gray_light_mask));
  litavg_green = sum(sum(image_double(:,:,2).*eroded_gray_light_mask)) / sum(sum(eroded_gray_light_mask));
  litavg_blue = sum(sum(image_double(:,:,3).*eroded_gray_light_mask)) / sum(sum(eroded_gray_light_mask));

  //REDUCTION - find averages
  ratio_red = litavg_red/shadowavg_red - 1;
  ratio_green = litavg_green/shadowavg_green - 1;
  ratio_blue = litavg_blue/shadowavg_blue - 1;

  //FinalMAP - use averages & current value to adjust
  result(:,:,1) = (ratio_red + 1)./((1-smoothmask)*ratio_red + 1).*image_double(:,:,1);
  result(:,:,2) = (ratio_green + 1)./((1-smoothmask)*ratio_green + 1).*image_double(:,:,2);
  result(:,:,3) = (ratio_blue + 1)./((1-smoothmask)*ratio_blue + 1).*image_double(:,:,3);
  */

#define CHANNELS 3 // we have 3 channels corresponding to RGB
__global__ void proc5(float * redData, float * greenData, float * blueData, float * resultImage, float * deviceSmoothOutputImageData , float* ratio_red, float* ratio_green, float* ratio_blue, int width, int height) {//this function creates result image and also reduces register trafic and has coalesced r g b reads
	
}
__global__ void proc5V1(float * inputImage, float * resultImage, float * deviceSmoothOutputImageData , float* ratio_red, float* ratio_green, float* ratio_blue, int width, int height) {//this function calculates result image and has coalesced r g b reads
	
}

#endif
