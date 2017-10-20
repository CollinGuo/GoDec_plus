%This is demo for background modeling by GoDec+. 
%Each frame of the video is reshaped into a column vector and all the
%vectors are concatenated to form the data matrix.

clear
video_name = 'Escalator';
video_dir = ['./Videos/' video_name '/'];
disp(video_name);
XO = [];
files = dir([video_dir,'*.bmp']);
fileName = fullfile(video_dir,files(1).name);
I = imread(fileName);
pixel_num = numel(I)/3;
I = rgb2gray(I);
select = randsample(pixel_num,round(pixel_num*0.02));
noise = randi([0,1],round(pixel_num*0.02),1)*255;
for i = 1:200
    fileName = fullfile(video_dir,files(i).name);
    I = imread(fileName);
    I = rgb2gray(I);
    I = double(I);
    XO = [XO,I(:)];
end


isize = size(I);

X = XO;
r = 2;

[m,n]=size(X);
tau = 20;

q = 0;
sigma = 1e+4;

epsilon = 1e-7;
tic
[L,RMSE2,~,~,iter]=lowrank_corr(X,r,sigma,epsilon,q);
toc
G = X - L ;


for i = 1:size(X,2)

    subplot(1,3,1);imagesc(reshape(X(:,i),isize));colormap(gray);axis image;axis off;title('Scene');
    subplot(1,3,2);imagesc(reshape(L(:,i),isize));colormap(gray);axis image;axis off;title('GoDec+:Background');
    subplot(1,3,3);imagesc(reshape(G(:,i),isize));colormap(gray);axis image;axis off;title('GoDec+:Foreground');
    pause(0.1)

end





