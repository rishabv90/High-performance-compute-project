 %I = imread('Dan7_0070.tif');
 %I = imread('plant_sd.png');
 %I = imread('plant_2.jpg');
 %I = imread('gf.png');
 I = imread('plt4.jpg');
       J = im2double(I);

      R = J(:,:,1);
      G = J(:,:,2);
      B = J(:,:,3);

     [len,wid] = size(R);
    
     % Generation L1 Chromaticity Image.
     for i = 1:len
        for j = 1:wid
           %if ((R(i,j)*G(i,j)*B(i,j))~= 0)
              l1(i,j) = R(i,j)/(R(i,j) + G(i,j) + B(i,j));
              l2(i,j) = G(i,j)/(R(i,j) + G(i,j) + B(i,j));
              l3(i,j) = B(i,j)/(R(i,j) + G(i,j) + B(i,j));
              
              ll1(i,j) = R(i,j)/(R(i,j) + 1 + B(i,j));
              ll2(i,j) = 1 /(R(i,j) + 1 + B(i,j));
              ll3(i,j) = B(i,j)/(R(i,j) + 1 + B(i,j));
           %else
            %  c1(i,j) = 1;
             % c2(i,j) = 1;
              %c3(i,j) = 1;
           %end
        end
     end
    l1cr = zeros(len,wid,3);
    l1cr(:,:,1)=l1;
    l1cr(:,:,2)=l2;
    l1cr(:,:,3)=l3;
    
    ll1cr = zeros(len,wid,3);
    ll1cr(:,:,1)=ll1;
    ll1cr(:,:,2)=ll2;
    ll1cr(:,:,3)=ll3;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

     % Generation of 2-D Log Chromaticity Image.
     for i = 1:len
        for j = 1:wid
           if ((R(i,j)*G(i,j)*B(i,j))~= 0)
              c1(i,j) = R(i,j)/((R(i,j)*G(i,j)*B(i,j))^(1/3));
              c2(i,j) = G(i,j)/((R(i,j)*G(i,j)*B(i,j))^(1/3));
              c3(i,j) = B(i,j)/((R(i,j)*G(i,j)*B(i,j))^(1/3));
           else
              c1(i,j) = 1;
              c2(i,j) = 1;
              c3(i,j) = 1;
        end
    end
end

rho1 = mat2gray(log(c1));
rho2 = mat2gray(log(c2));
rho3 = mat2gray(log(c3));

X1 = mat2gray(rho1*1/(sqrt(2)) - rho2*1/(sqrt(2))); %(1/sqrt(2); -1/sqrt(2); 0)
X2 = mat2gray(rho1*1/(sqrt(6)) + rho2*1/(sqrt(6)) - rho3*2/(sqrt(6))); %(1/sqrt(6); 1/sqrt(6); -2/sqrt(6))

theta = 120;

%InvariantImage = cos(theta*pi/180)*X1 + sin(theta*pi/180)*X2;

X3 = (rho1*1/(sqrt(2)) - rho2*1/(sqrt(2))); 
X4 = (-rho1*sqrt(6)/4 + rho2*sqrt(6)/4 + rho3*(sqrt(6)/4)); 

iimg = cos(theta*pi/180)*X3 + sin(theta*pi/180)*X4;

iimg2 = cos(theta*pi/180)*X1 + sin(theta*pi/180)*X2;

%X1 = mat2gray(rho1*1/(sqrt(2)) - rho2*1/(sqrt(2))); 
%X2 = mat2gray(-rho1*sqrt(6)/4 + rho2*sqrt(6)/4 + rho3*(sqrt(6)/4)); 
for t=1:180

    delta = t*pi/180;
    img = cos(delta)*X1 + sin(delta)*X2 ;
    
    logimg = -log(img);
    r=img.*logimg; 
    
    n(t)=mean(abs(r(:)));
end;

figure
plot(abs(n(:)), 'DisplayName', 'n', 'YDataSource', 'n'); figure(gcf)
[v,idx]=min(n(:));

