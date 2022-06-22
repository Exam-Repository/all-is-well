# Spectrun Analysis
```
fs=8000;
x2=10*sin(2*pi*(f2/fs)*n);
subplot(2,2,2);
plot(n,x2);
axis([0,N,-12,12]),grid;
xlabel('sample number');
ylabel('amplitude');
title('sinewave[f2=2KHz,fs=8KHz]');
y=x1+x2;
fft(y);
subplot(2,2,3);
plot(n,y);
axis([0,N,-12,12]),grid;
xlabel('sample number');
ylabel('amplitude');
title('complex signal');
z=abs(fft(y));
subplot(2,2,4);
plot(z);
xlabel('sample number');
ylabel('amplitude');
title('absolute fft');

# 2-IIR FILTER USING BUTTERWORTH TECHNIQUE:
```
clc;
clear all;
close all;
fpb=1000;
fsb=2000;
rp=0.2;
rs=50;
fs=10000;
fpb1=[1000 2000];
fsb1=[500 2500];
M=512;
wp=2*fpb/fs;
ws=2*fsb/fs;
wp1=2*fpb1/fs;
ws1=2*fsb1/fs;
%low pass
%option1
[N,wn]=buttord(wp,ws,rp,rs);
[b,a]=butter(N,wn);
figure,freqz(b,a,M,fs);
title('freq res LP')
%option2
w=0:0.01:pi;
h1=freqz(b,a,w);
figure(2);
plot(abs(h1));
xlabel('frequency in hz');
ylabel('amplitude');
title('low pass');
%high pass
%option1
%option1
[N,wn]=buttord(wp,ws,rp,rs);
[b,a]=butter(N,wn,'high');
figure(3);
freqz(b,a,M,fs);
title('freq res HP')
%option2
w=0:0.01:pi;
h1=freqz(b,a,w);
figure(4);
plot(abs(h1));
xlabel('frequency in hz');
ylabel('amplitude');
title('high pass')
%band pass
%option1
[N,wn1]=buttord(wp1,ws1,rp,rs);
[b,a]=butter(N,wn1);
figure(5);
freqz(b,a,M,fs);
title('freq res BP')
%option2
w=0:0.01:pi;
h2=freqz(b,a,w);
figure(6);
plot(abs(h2));
xlabel('frequency in hz');
ylabel('amplitude');
title('band pass');
%band stop
%option1
[N,wn1]=buttord(wp1,ws1,rp,rs);
[b,a]=butter(N,wn1,'stop');
figure(7);
freqz(b,a,M,fs);
title('freq res BS')
%option2
w=0:0.01:pi;
h2=freqz(b,a,w);
figure(8);
plot(abs(h2));
xlabel('frequency in hz');
ylabel('amplitude');
title('band stop');

# 3-IIR FILTER USING CHEBYSHEV TECHNIQUE
```
clc;
clear all;
close all;
fpb=1000;
fsb=2000;
rp=0.2;
rs=50;
fs=10000;
fpb1=[1000 2000];
fsb1=[500 2500];
M=512;
wp=2*fpb/fs;
ws=2*fsb/fs;
wp1=2*fpb1/fs;
ws1=2*fsb1/fs;
%LPF
%option 1
[N,wn]=cheb1ord(wp,ws,rp,rs);
[b,a]=cheby1(N,rp,wn);
figure,freqz(b,a,M,fs);
%option 2 
w=0:0.01:pi;
h1=freqz(b,a,w);
figure(2);
% option3
plot(abs(h1));
% HPF
% option 1 
[N,wn1]=cheb1ord(wp,ws,rp,rs);
[c,d]=cheby1(N,rp,wn1,'high');
figure,freqz(c,d,M,fs);
% option 2
w=0:0.01:pi;
h2=freqz(c,d,w);
figure(4);
plot(abs(h2));
%BPF
[N,wn2]=cheb1ord(wp1,ws1,rp,rs);
[e,f]=cheby1(N,rp,wn2);
figure,freqz(e,f,M,fs);
w=0:0.01:pi;
h3=freqz(e,f,w);
figure(6);
plot(abs(h3));
%BSF
[N,wn3]=cheb1ord(wp1,ws1,rp,rs);
[g,h]=cheby1(N,rp,wn3,'stop');
figure,freqz(g,h,M,fs);
w=0:0.01:pi;
h4=freqz(g,h,w);
figure(8);
plot(abs(h4));

# -MULTI-RATE FILTER
```
clc;
clear all;
close all;
N=64;
n=0:1:N;
f=1000;
fs=12000;
x1=10*sin((2*pi*(f/fs)*n));
N=64;
n=0:1:N;
f=4000;
x2=10*sin((2*pi*(f/fs)*n));
xs=x1+x2;
subplot(2,2,1);
stem(xs);
xlabel('Sample number');
ylabel('Amplitude');
title('Complex signal');
fpb=1000;
fsb=2200;
rp=0.2;
rs=45;
num=20*log10(sqrt(rp*rs))-13;
dem=14.6*(fsb-fpb)/fs;
n=ceil(num/dem);
n=abs(n);
wp=2*fpb/fs;
ws=2*fsb/fs;
wn=(ws+wp)/2;
if(rem(n,2)==0)
 m=n+1;
else
 m=n;
 n=n-1;
end;
w=hamming(m);
hn=fir1(n,wn,'low',w); 
olpf=conv(xs,hn);
subplot(2,2,2);
stem(olpf);
xlabel('Sample number');
ylabel('Amplitude');
title('OLPF');
dolpf=downsample(olpf,2);
subplot(2,2,3);
stem(dolpf);
xlabel('Sample number');
ylabel('Amplitude');
title('DOLPF');
xint=upsample(xs,2);
uolpf=conv(xint,hn);
subplot(2,2,4);
stem(uolpf);
xlabel('Sample number');
ylabel('Amplitude');
title('UOLPF');

# kaiser
```
PROGRAM: Kaiser window: Low pass
clc;
clear all;
close all;
fsb=2000;
fpb=3200;
fs=10000;
rp=0.2;
rs=60;
nr=-20*log10(sqrt(rp*rs))-13;
dr=14.6*((fsb-fpb)/fs);
N=nr/dr;
N=ceil(N);
if(rem(N,2)==0)
M=N+1;
else
 M=N;
 N=N-1;
end
wsb=2*(fsb/fs);
wpb=2*(fpb/fs);
w=kaiser(M);
%Low Pass Filter
wn1=(wsb+wpb)/2;
b1=fir1(N,wn1,w);
%option1
figure, freqz(b1,1,1000,fs);
%Option2
a=freqz(b1,1,1000,fs);
figure,plot(abs(a));
```