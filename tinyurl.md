```
Experiment 1 

Sampling (spatial)

clc;
clear all;
close all;
a = imread('peppers.png');
b = rgb2gray(a);
figure();
imshow(b);
title('Normal gray');
c= imresize(b,2);
figure();
imshow(c);
title('Double');
d= imresize(b,0.5);
figure();
imshow(d);
title('Half');
f= imresize(b,[100,100]);
figure();
imshow(f);
title('100*100');

Quantisation

clc;
clear all;
close all;
a = imread('peppers.png');
b = rgb2gray(a);
subplot(2,3,1);
imshow(a);
title('Original');
c = grayslice(a,128);
subplot(2,3,2);
imshow(c);
title('128');
d = grayslice(a,64);
subplot(2,3,3);
imshow(d);
title('64');
e = grayslice(a,32);
subplot(2,3,4);
imshow(e);
title('32');
f = grayslice(a,16);
subplot(2,3,5);
imshow(f);
title('16');
g = grayslice(a,8);
subplot(2,3,6);
imshow(g);
title('8');

Experiment 2

Gamma transform

clc;
clear all;
close all;
a=imread('cameraman.tif');
subplot(2,4,1);
imshow(a);
title('Original');
c=1;
g1=2;
d=im2double(a);
s1=c*(d.^g1);
subplot(2,4,2);
imshow(s1);
title('g1=2');
g2=5;
s2=c*(d.^g2);
subplot(2,4,3);
imshow(s2);
title('g2=5');
g3=10;
s3=c*(d.^g3);
subplot(2,4,4);
imshow(s3);
title('g3=10');
g4=0.1;
s4=c*(d.^g4);
subplot(2,4,5);
imshow(s4);
title('g4=0.1');
g5=0.2;
s5=c*(d.^g5);
subplot(2,4,6);
imshow(s5);
title('g5=0.2');
g6=0.3;
s6=c*(d.^g6);
subplot(2,4,7);
imshow(s6);
title('g6=0.3');
g7=0.5;
s7=c*(d.^g7);
subplot(2,4,8);
imshow(s7);
title('g7=0.5');

Log transform

clc;
clear all;
close all;
a=imread('cameraman.tif');
subplot(2,4,1);
imshow(a);
title('Original');
c=1;
g1=2;
d=im2double(a);
s1=c*(d.^g1);
subplot(2,4,2);
imshow(s1);
title('g1=2');
g2=5;
s2=c*(d.^g2);
subplot(2,4,3);
imshow(s2);
title('g2=5');
g3=10;
s3=c*(d.^g3);
subplot(2,4,4);
imshow(s3);
title('g3=10');
g4=0.1;
s4=c*(d.^g4);
subplot(2,4,5);
imshow(s4);
title('g4=0.1');
g5=0.2;
s5=c*(d.^g5);
subplot(2,4,6);
imshow(s5);
title('g5=0.2');
g6=0.3;
s6=c*(d.^g6);
subplot(2,4,7);
imshow(s6);
title('g6=0.3');
g7=0.5;
s7=c*(d.^g7);
subplot(2,4,8);
imshow(s7);
title('g7=0.5');

Experiment 3 

Histogram with built-in

clc;
close all;
a = imread('peppers.png');
a = rgb2gray(a);
subplot(2,2,1);
imshow(a);
title('Original Image');
subplot(2,2,2);
imhist(a);
title('Histogram of the Image');
subplot(2,2,3);
histeq(a);
title('Equalized Histogram');
subplot(2,2,4);
b = adapthisteq(a);
imhist(b);
title('Adapted Equalized Histogram');

Histogram without built-in

clc;
close all;
a = imread('cameraman.tif');
subplot(2,1,1);
imshow(a);
title('Original Image');
subplot(2,1,2);
z = im2double(a);
k = zeros(256,1);
for i=1:size(z,1)
    for j=1:size(z,1)
        for k1=0:255
            if a(i,j)==k1
                k(k1+1) = k(k1+1)+1;
            end
        end
    end
end
bar(k)

Experiment 4 

Kernel
 
clc;
clear all;
close all;
b=imread('cameraman.tif');
subplot(2,3,1);
imshow(b);
subplot(2,3,2);
imshow(b);
a=im2double(b);
[N N]=size(a);
[k n]=meshgrid(0:N-1,0:N-1);
ker=(exp(-(j*(2*pi*(n.*k))/N)))/sqrt(N);
trans1=ker*a*ker;
d=fftshift(abs(trans1));
subplot(2,3,3);
imshow(d);
invker=(exp(j*2*pi*(n.*k)/N))/sqrt(N);
recon=invker*trans1*invker;
subplot(2,3,4);
imshow(recon);
for i=1:N
for j=1:N
    error(i,j)=(a(i,j)-recon(i,j))^2;
end
end
error1=sum(sum(error))/(N*N);
subplot(2,3,5);
imshow(error1);

DCT 

clc;
close all;
clear all;
a=imread('peppers.png');
a = imresize(a,[256 256]);
b = rgb2gray(a);
subplot(2,2,1);
imshow(b)
title('original');
c=dct2(b);
subplot(2,2,2);
imshow(c,[0 255])
title('dctimg');
d=idct2(c);
subplot(2,2,3);
imshow(d, [0 255])
title('idctimg');
[N N]=size(b)
for i=1:N
    for j=1:N
        error(i,j)=(b(i,j)-d(i,j))^2;
    end
end
error1=sum(sum(error))/(N*N);
subplot(2,2,4);
imshow(error1);
title('error');

Haar 

clc;
clear all;
close all;
in=imread('peppers.png');
subplot(1,3,1);
imshow(in);
title('Original image');
[a,b,c,d]=dwt2(in,'haar');
p=[uint8(a),b;c,d]
subplot(1,3,2);
imshow(p);
title('First level decomposition');
[a1,a2,a3,a4]=dwt2(a,'haar');
p1=[a1,a2;a3,a4];
p2=[uint8(p1),b;c,d]
subplot(1,3,3);
imshow(p2);
title('Second level decomposition'); 

Experiment 5 

a =imread('peppers.png');
subplot(2,3,1);
imshow(a);
title('original image');
a=rgb2gray(a);
b = imnoise(a,'salt & pepper')
subplot(2,3,2);
imshow(b);
title('noise 1-salt & pepper');
b = imnoise(a,'speckle')
subplot(2,3,3);
imshow(b);
title('noise 2-speckle');
c = medfilt2(b);
subplot(2,3,5);
imshow(c);
title('non linear filter');
d=fspecial('average');
subplot(2,3,4);
imshow(d);
title('box filter');
e=imfilter(b,d,'conv');
subplot(2,3,6);
imshow(e);
title('linear filter'); 

Experiment 6

Frequency domain HPF 

clc;
clear all;
close all;
im=imread('peppers.png');
subplot(2,3,1);
imshow(im);
title('org')
im=rgb2gray(im);
ftim=fft2(im);
cftim=fftshift(ftim);
subplot(2,3,2);
imshow(cftim);
title('fft of org');
D0=20;
n=2;
[M,N]=size(im);
cx=round(M/2);
cy=round(N/2);
H=zeros(M,N);
for i=1:M
    for j=1:N
        d=sqrt((i-cx).^2+(j-cy).^2);
        H(i,j)=1/(1+((D0/d).^(2*n)));
    end;
end;
%H1=3+log10(double(H));
subplot(2,3,3);
imshow(H);
title('H');
%H=H./max(H(:));

fftim=cftim.*H;
subplot(2,3,4);
imshow(fftim);
title('fftim');
sim=abs(ifft2(fftim));
subplot(2,3,5);
imshow(uint8(sim));
title('sim');

Frequency domain LPF 

clc;
clear all;
close all;
im=imread('peppers.png');
subplot(2,3,1);
imshow(im);
title('original image');
im=rgb2gray(im);
ftim=fft2(im);
cftim=fftshift(ftim);
subplot(2,3,2);
imshow(cftim);
title('fourier transform');
D0=20;
n=2;
[M,N]=size(im);
Cx=round(M/2);
Cy=round(N/2);
H=zeros(M,N);
for i=1:M
    for j=1:N
       d=sqrt((i-Cx).^2+(j-Cy).^2);
       H(i,j)=1/(1+((d/D0).^(2*n)));
    end;
end;
subplot(2,3,3);
imshow(H);
title('H(i,j)');
fftim=cftim.*H;
subplot(2,3,4);
imshow(fftim);
title('fft');
sim=abs(ifft2(fftim));
sim=uint8(sim);
subplot(2,3,5);
imshow(sim);
title('LPF'); 

Experiment 7 

clc;
clear all;
close all;
in=imread('cameraman.tif');
subplot(6,3,2);
imshow(in);
title('Original image');
a=edge(in,'prewitt');
subplot(6,3,4);
imshow(a);
title('Prewitt:Default');
a1=edge(in,'prewitt','horizontal');
subplot(6,3,5);
imshow(a1);
title('Prewitt:Horizontal');
a2=edge(in,'prewitt','vertical');
subplot(6,3,6);
imshow(a2);
title('Prewitt:Vertical');
b=edge(in,'sobel');
subplot(6,3,7);
imshow(b);
title('Sobel:Default');
b1=edge(in,'sobel','horizontal');
subplot(6,3,8);
imshow(b1);
title('Sobel:Horizontal');
b2=edge(in,'sobel','vertical');
subplot(6,3,9);
imshow(b2);
title('Sobel:Vertical');
c=edge(in,'roberts');
subplot(6,3,10);
imshow(c);
title('Roberts:Default');
c1=edge(in,'roberts','horizontal');
subplot(6,3,11);
imshow(c1);
title('Roberts:Horizontal');
c2=edge(in,'roberts','vertical');
subplot(6,3,12);
imshow(c2);
title('Roberts:Vertical');
d=edge(in,'zerocross');
subplot(6,3,13);
imshow(d);
title('Zerocross:Default');
d1=edge(in,'zerocross','horizontal');
subplot(6,3,14);
imshow(d1);
title('Zerocross:Horizontal');
d2=edge(in,'zerocross','vertical');
subplot(6,3,15);
imshow(d2);
title('Zerocross:Vertical');
e=edge(in,'canny');
subplot(6,3,16);
imshow(e);
title('Canny:Default');
e1=edge(in,'canny','horizontal');
subplot(6,3,17);
imshow(e1);
title('Canny:Horizontal');
e2=edge(in,'canny','vertical');
subplot(6,3,18);
imshow(e2);
title('Canny:Vertical'); 

Experiment 8 

clc;
clear all;
close all;
im = imread('pout.tif');
se1 = strel('arbitrary',[1 1 1 ; 0 1 0 ; 1 1 1]);
se2 = strel('line',20,0);
se3 = strel('square', 10);
se4 = strel('rectangle',[20 40]);
se5 = strel('disk', 20);
se6 = strel('ball', 5, 4);
figure(1);
subplot(2,4,1);
imshow(im);
title('Org');
subplot(2,4,2);
imshow(imdilate(im, se1));
title('Dilate : Arbitrary');
subplot(2,4,3);
imshow(imdilate(im, se2));
title('Dilate : Line');
subplot(2,4,4);
imshow(imdilate(im, se3));
title('Dilate : Square');
subplot(2,4,5);
imshow(imdilate(im, se4));
title('Dilate : Rectangle');
subplot(2,4,6);
imshow(imdilate(im, se5));
title('Dilate : Disk');
subplot(2,4,7);
imshow(imdilate(im, se6));
title('Dilate : Ball');
figure(2);
subplot(2,4,1);
imshow(im);
title('Org');
subplot(2,4,2);
imshow(imerode(im, se1));
title('Erode : Arbitrary');
subplot(2,4,3);
imshow(imerode(im, se2));
title('Erode : Line');
subplot(2,4,4);
imshow(imerode(im, se3));
title('Erode : Square');
subplot(2,4,5);
imshow(imerode(im, se4));
title('Erode : Rectangle');
subplot(2,4,6);
imshow(imerode(im, se5));
title('Erode : Disk');
subplot(2,4,7);
imshow(imerode(im, se6));
title('Erode : Ball');
figure(3);
subplot(2,4,1);
imshow(im);
title('Org');
subplot(2,4,2);
imshow(imopen(im, se1));
title('Open : Arbitrary');
subplot(2,4,3);
imshow(imopen(im, se2));
title('Open : Line');
subplot(2,4,4);
imshow(imopen(im, se3));
title('Open : Square');
subplot(2,4,5);
imshow(imopen(im, se4));
title('Open : Rectangle');
subplot(2,4,6);
imshow(imopen(im, se5));
title('Open : Disk');
subplot(2,4,7);
imshow(imopen(im, se6));
title('Open : Ball');
figure(4);
subplot(2,4,1);
imshow(im);
title('Org');
subplot(2,4,2);
imshow(imclose(im, se1));
title('Close : Arbitrary');
subplot(2,4,3);
imshow(imclose(im, se2));
title('Close : Line');
subplot(2,4,4);
imshow(imclose(im, se3));
title('Close : Square');
subplot(2,4,5);
imshow(imclose(im, se4));
title('Close : Rectangle');
subplot(2,4,6);
imshow(imclose(im, se5));
title('Close : Disk');
subplot(2,4,7);
imshow(imclose(im, se6));
title('Close : Ball');

Experiment 9 

Colour planes 

clc;
clear all;
close all;
a=imread('football.jpg');
subplot(2,2,1);
imshow(a);
title('original');
[r c d]=size(a);
z=zeros(r,c);
temp=a;
temp(:,:,2)=z; temp(:,:,3)=z;
subplot(2,2,2);
imshow(temp);
title('red');
z1=zeros(r,c);
temp=a;
temp(:,:,1)=z1; temp(:,:,3)=z1;
subplot(2,2,3);
imshow(temp);
title('green');
z2=zeros(r,c);
temp=a;
temp(:,:,1)=z1; temp(:,:,2)=z1;
subplot(2,2,4);
imshow(temp);
title('blue');

Colour models 

clc;
clear all;
close all;
a=imread('football.jpg');
subplot(2,2,1);
imshow(a);
title('original');
b=rgb2ntsc(a);
subplot(2,2,2);
imshow(b);
title('ntsc');
c=rgb2hsv(a);
subplot(2,2,3);
imshow(c);
title('hsv');
d=rgb2ycbcr(a);
subplot(2,2,4);
imshow(d);
title('ycbcr');
```