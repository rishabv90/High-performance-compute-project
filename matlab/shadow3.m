% Serial implementation of third shadow removal algorithm
% taken from "Contrast Limited Adaptive Histogram Equalization (CLAHE) 
% and Shadow Removal for Controlled Environment Plant Production Systems"
% By Burak Unal

clear
close all
format compact


%% Reading and plotting original image
image = imread('../Data/plt4.jpg');
figure('NumberTitle','off','Name','Original Lettuce Image');
imshow(image)
title('Original Lettuce Lmage');

tic
%% Calculating Color Invariant Image
image_double = im2double(image);
redPart = image_double(:,:,1);
greenPart = image_double(:,:,2);
bluePart = image_double(:,:,3);

[row, col, ~] = size(image);
color_invariant_image = zeros(row,col,3);
for i=1:row
    for j=1:col
        color_invariant_image(i,j,1) = atan(redPart(i,j)/max(greenPart(i,j),bluePart(i,j)));
        color_invariant_image(i,j,2) = atan(greenPart(i,j)/max(redPart(i,j),bluePart(i,j)));
        color_invariant_image(i,j,3) = atan(bluePart(i,j)/max(redPart(i,j),greenPart(i,j)));
    end
end
figure('NumberTitle','off','Name','Color Invariant Image');
imshow(color_invariant_image);
title("Color Invariant Image");

%% Calculating RGB -> YUV and creating mask using Otsu's method
yuv = rgb2ycbcr(image);
yuv_mask  = double(imbinarize(yuv(:,:,2), eddie_graythresh(yuv(:,:,2))));
figure('NumberTitle','off','Name','YUV Image');
imshow(yuv);
title("YUV Image");

figure('NumberTitle','off','Name','YUV Mask Image');
imshow(yuv_mask);
title("YUV Mask Image");
%% Calculating Color Invariant RGB -> Grayscale and creating mask using Otsu's method
gray = rgb2gray(color_invariant_image);
gray_mask  = 1-double(imbinarize(gray, eddie_graythresh(gray)));
figure('NumberTitle','off','Name','Gray Image');
imshow(gray);
title("Gray Image");

figure('NumberTitle','off','Name','Gray Mask Image');
imshow(gray_mask);
title("Gray Mask Image");
%% Defining structuring element and performing erosion
strel = [1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1];
eroded_gray_shadow_mask = imerode(gray_mask, strel);
eroded_gray_light_mask = imerode(1-gray_mask, strel);
figure('NumberTitle','off','Name','Eroded Gray Mask Image');
imshow(eroded_gray_shadow_mask);
title("Eroded Gray Shadow Mask Image");

figure('NumberTitle','off','Name','Eroded Gray Light Mask Image');
imshow(eroded_gray_light_mask);
title("Eroded Gray Light Mask Image");
%% Using structuring element to create smooth mask
strel = strel/25;
smoothmask = conv2(yuv_mask, strel, 'same');
figure('NumberTitle','off','Name','Smooth Mask Image');
imshow(smoothmask);
title("Smooth Mask Image");
%% Finding average channel values in shadow/light areas for every channel
shadowavg_red = sum(sum(image_double(:,:,1).*eroded_gray_shadow_mask)) / sum(sum(eroded_gray_shadow_mask));
shadowavg_green = sum(sum(image_double(:,:,2).*eroded_gray_shadow_mask)) / sum(sum(eroded_gray_shadow_mask));
shadowavg_blue = sum(sum(image_double(:,:,3).*eroded_gray_shadow_mask)) / sum(sum(eroded_gray_shadow_mask));
litavg_red = sum(sum(image_double(:,:,1).*eroded_gray_light_mask)) / sum(sum(eroded_gray_light_mask));
litavg_green = sum(sum(image_double(:,:,2).*eroded_gray_light_mask)) / sum(sum(eroded_gray_light_mask));
litavg_blue = sum(sum(image_double(:,:,3).*eroded_gray_light_mask)) / sum(sum(eroded_gray_light_mask));
  
%% Calculating ratio of light-to-shadow in every channel
ratio_red = litavg_red/shadowavg_red - 1;
ratio_green = litavg_green/shadowavg_green - 1;
ratio_blue = litavg_blue/shadowavg_blue - 1;


%% Removing shadow
result = zeros(size(image_double));
result(:,:,1) = (ratio_red + 1)./((1-smoothmask)*ratio_red + 1).*image_double(:,:,1);
result(:,:,2) = (ratio_green + 1)./((1-smoothmask)*ratio_green + 1).*image_double(:,:,2);
result(:,:,3) = (ratio_blue + 1)./((1-smoothmask)*ratio_blue + 1).*image_double(:,:,3);
toc

%% Outputing final result
figure('NumberTitle','off','Name','Shadowless Lettuce Image');
imshow(result);
title("Shadowless Lettuce Image");    