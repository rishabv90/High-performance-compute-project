% this source code performs shadow removal
% clear
% close all

%frame1 = imread('plant_sd.png')
image = imread('plt4.jpg');
%image = imread('plt7_300_400.jpg');
%image = imread('plt6_1280_720.jpg');
%frame1 = imread('plt_5000.jpg');
%frame1 = imread('plt6_1280_720.jpg');
%frame1 = imread('plant_2.jpg');
%frame1 = imread('Dan7_0070.tif');
%frame1 = imread('image001.jpg');

% computing size of the image
s_im = size(image);
%tic  


%this performs Color Incariance algorithm
%frame1 = imread('plant_sd.png');
frame1 = image; 
figure
imshow(frame1), title('Original Lettuce Lmage');

% Point to the object with the shadow
[xin,yin]=ginput(2) % get one input "ginput(2)" gets two points...
xin = round(xin);
yin = round(yin);
%tic

% get the cusen part of the image
%im1 = frame1(yin(1):yin(2),xin(1):xin(2),:);
im1 = frame1;
figure
imshow(im1);


% start to calculation for Colr Invariant image
im1 = im2double(im1);
redPart = im1(:,:,1);
greenPart = im1(:,:,2);
bluePart = im1(:,:,3);

% show the R/G image
figure
colormap(gray);
imshow(redPart./greenPart);
gry = redPart./greenPart;
%imshow(bluePart./greenPart);

% show the B/G
%subplot(2,2,4);
%colormap(gray);
%imshow(bluePart./greenPart);

% calculation color invarince image
[row col three] = size(im1);
imm = zeros(row,col,3);

for i=1:row
    for j=1:col
        imm(i,j,1) = atan(redPart(i,j)/max(greenPart(i,j),bluePart(i,j)));
        imm(i,j,2) = atan(greenPart(i,j)/max(redPart(i,j),bluePart(i,j)));
        imm(i,j,3) = atan(bluePart(i,j)/max(redPart(i,j),greenPart(i,j)));
    end
end
%toc
%colormap(gray);
%imagesc(imm);
figure
imshow(imm)

% RBG to YUV convertion
yuv = rgb2ycbcr(image);
gray = rgb2gray(imm);

% this part represnts the formula for shadow removal 
% thi is the part you will change the coefficint factors
% treshold setting is here
mask2  = 1-double(im2bw(gray, (graythresh(gray))));
%mask  = 1-double(im2bw(iimg2, (graythresh(iimg2)))); %-0.02)));
%mask  = 1-double(im2bw(iimg2, (graythresh(iimg2)-0.01)));
mask  = double(im2bw(yuv(:,:,2), (graythresh(yuv(:,:,2)-0.5))));


% playing with shadow intensity
% this shows the intensity of shadow 
% Color Invariance image
bl = imm(:,:,3);
bl = bl .* bl;
ratio2 = bl .* bl;

% YUV space
% bl = yuv(:,:,1);
% bl = im2double(bl);
% ratio2 = bl; % .* bl;


%shadow_core = ttt;

    image = im2double(image);
    % structuring element for the shadow mask buring, and the shadow/light
    % core detection
    strel = [0 1 1 1 0; 1 1 1 1 1; 1 1 1 1 1; 1 1 1 1 1; 0 1 1 1 0];
    
    % computing shadow/light  core (pixels not on the blured adge of the
    % shadow area)
    shadow_core = mask; %imerode(mask, strel);
    lit_core = 1-mask; %imerode(1-mask, strel);
    
    shadow_core2 = imerode(mask2, strel);
    lit_core2 = imerode(1-mask2, strel);
    
    % smoothing the mask
    smoothmask = conv2(mask, strel/21, 'same');
        
    % averaging pixel intensities in the shadow/lit areas
    shadowavg_red = sum(sum(image(:,:,1).*shadow_core2)) / sum(sum(shadow_core2));
    shadowavg_green = sum(sum(image(:,:,2).*shadow_core2)) / sum(sum(shadow_core2));
    shadowavg_blue = sum(sum(image(:,:,3).*shadow_core2)) / sum(sum(shadow_core2));

    litavg_red = sum(sum(image(:,:,1).*lit_core2)) / sum(sum(lit_core2));
    litavg_green = sum(sum(image(:,:,2).*lit_core2)) / sum(sum(lit_core2));
    litavg_blue = sum(sum(image(:,:,3).*lit_core2)) / sum(sum(lit_core2));
    
    result = image;
  
 % implementation of the different shadow removals
 % the first one
        
        % compiting colour difference between the shadow/lit areas
        %diff_red = (litavg_red - shadowavg_red) / 2 ;
        diff_red = litavg_red - shadowavg_red;
        diff_green = litavg_green - shadowavg_green;
        diff_blue = litavg_blue - shadowavg_blue;
        
        % play with ratio2
        result(:,:,1) = image(:,:,1) + smoothmask .* ratio2; %bl .*0.4; %diff_red;
        result(:,:,2) = image(:,:,2) + smoothmask .* ratio2; %bl .*0.4; %diff_green;
        result(:,:,3) = image(:,:,3) + smoothmask .* ratio2; %bl .*0.4; %diff_blue;
        figure
        imshow(result);

result = image;
% the second one
        
        % computing ratio of shadow/lit area luminance
        ratio_red = litavg_red/shadowavg_red;
        ratio_green = litavg_green/shadowavg_green;
        ratio_blue = litavg_blue/shadowavg_blue;
        
        
        %play with ratio_red/green/blue
        result(:,:,1) = image(:,:,1).*(1-mask2) + mask.*ratio_red.*image(:,:,1).* ratio2;
        result(:,:,2) = image(:,:,2).*(1-mask2) + mask.*ratio_green.*image(:,:,2).* ratio2;
        result(:,:,3) = image(:,:,3).*(1-mask2) + mask.*ratio_blue.*image(:,:,3).* ratio2;
        
        %toc
        figure
        imshow(result);
        %tic  
result = image;
 
        
        % computing ratio of the luminances of the directed, and global
        % lights
        ratio_red = litavg_red/shadowavg_red - 1;
        ratio_green = litavg_green/shadowavg_green - 1;
        ratio_blue = litavg_blue/shadowavg_blue - 1;
        
        % appliing shadow removal formula
        % (too long for the comment -> see documentation :) )
        result(:,:,1) = (ratio_red + 1)./((1-smoothmask)*ratio_red + 1).*image(:,:,1);
        result(:,:,2) = (ratio_green + 1)./((1-smoothmask)*ratio_green + 1).*image(:,:,2);
        result(:,:,3) = (ratio_blue + 1)./((1-smoothmask)*ratio_blue + 1).*image(:,:,3);
        %toc
        figure
        imshow(result);
        %tic
result = image;

        % ycbcr colourspace
        
        % conversion to ycbcr
        ycbcr = rgb2ycbcr(image);
        
        % computing averade channel values in ycbcr space
        shadowavg_y = sum(sum(ycbcr(:,:,1).*shadow_core2)) / sum(sum(shadow_core2));
        shadowavg_cb = sum(sum(ycbcr(:,:,2).*shadow_core2)) / sum(sum(shadow_core2));
        shadowavg_cr = sum(sum(ycbcr(:,:,3).*shadow_core2)) / sum(sum(shadow_core2));

        litavg_y = sum(sum(ycbcr(:,:,1).*lit_core2)) / sum(sum(lit_core2));
        litavg_cb = sum(sum(ycbcr(:,:,2).*lit_core2)) / sum(sum(lit_core2));
        litavg_cr = sum(sum(ycbcr(:,:,3).*lit_core2)) / sum(sum(lit_core2));
        
        % computing ratio, and difference in ycbcr space
        diff_y = litavg_y - shadowavg_y;
        diff_cb = litavg_cb - shadowavg_cb;
        diff_cr = litavg_cr - shadowavg_cr;

        ratio_y = litavg_y/shadowavg_y;
        ratio_cb = litavg_cb/shadowavg_cb;
        ratio_cr = litavg_cr/shadowavg_cr;

        % shadow correction, see formulas above
        % y channel has an additive correction
        % cb, and cr channels gets a model based correction
        res_ycbcr = ycbcr;
        res_ycbcr(:,:,1) = ycbcr(:,:,1) + mask .* ratio2; %diff_y;
        res_ycbcr(:,:,2) = ycbcr(:,:,2).*(1-mask) + mask.*ratio_cb.*ycbcr(:,:,2);
        res_ycbcr(:,:,3) = ycbcr(:,:,3).*(1-mask) + mask.*ratio_cr.*ycbcr(:,:,3);
        
        % conversion back to rgb colourspace
        result = ycbcr2rgb(res_ycbcr);
        %toc
        figure
        imshow(result);
    