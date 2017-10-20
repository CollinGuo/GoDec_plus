%This is demo for shadow/illumination removement by GoDec+.
%Each image is reshaped into a column vector and all the
%vectors are concatenated to form the data matrix.
clear

load('yaleB_cropped.mat');
isize = [192,168];
r = 3;


sigma = 1e+4;

q = 0;

noise_type = 1:9;
for k = 1:38
    for n_i = 1:length(noise_type)
        data = cell2mat(dataset(k));
        
        X = data;
        switch (noise_type(n_i))
            case 1
                disp('Gaussian noise1')
                X = X+normrnd(0,50,size(X));
            case 2
                disp('Gaussian noise2')
                X = X+normrnd(0,30,size(X));
            case 3
                disp('Gaussian noise3')
                X = X+normrnd(100,30,size(X));
            case 4
                disp('Gaussian noise4')
                X = X+normrnd(100,50,size(X));
            case 5
                disp('Laplacian noise')
                noise = rand(size(X));
                noise = 30*sign(0.5-noise).*(1/sqrt(2)).*log(2*min(noise,1-noise));
                X = X+noise;
            case 6
                disp('Salt & Pepper noise1')
                salt_pepper_rate=0.2;
                pixel_num = size(X,1);
                for i = 1:size(X,2)
                    select = randsample(pixel_num,round(pixel_num*salt_pepper_rate));
                    noise = randi([0,1],round(pixel_num*salt_pepper_rate),1)*255;
                    I = X(:,i);
                    I(select) = noise;
                    X(:,i) = I;
                end
            case 7
                disp('Salt & Pepper noise2')
                salt_pepper_rate=0.15;
                pixel_num = size(X,1);
                for i = 1:size(X,2)
                    select = randsample(pixel_num,round(pixel_num*salt_pepper_rate));
                    noise = randi([0,1],round(pixel_num*salt_pepper_rate),1)*255;
                    I = X(:,i);
                    I(select) = noise;
                    X(:,i) = I;
                end
            case 8
                disp('No extra noise')
            case 9
                disp('occlusion')
                occ_size = 60;
                for j = 1:size(X,2)
                    I = X(:,j);
                    I = reshape(I,isize);
                    x = randi(isize(2)-(occ_size-1),1);
                    y = randi(isize(1)-(occ_size-1),1);
                    I(y:y+occ_size-1,x:x+occ_size-1)=0;
                    I = I(:);
                    X(:,j) = I;
                end
        end
        
        epsilon = 1e-7;
        
        
        tic
        [L,RMSE,base,Q,iter]=lowrank_corr(X,r,sigma,epsilon,q);
        toc
        G = X-L;
        for i = 1:size(X,2)
            subplot(1,3,1);imagesc(reshape(X(:,i),isize));colormap(gray);axis image;axis off;title('X(Sample)');
            subplot(1,3,2);imagesc(reshape(L(:,i),isize));colormap(gray);axis image;axis off;title('L(Low-rank)');
            subplot(1,3,3);imagesc(reshape(G(:,i),isize));colormap(gray);axis image;axis off;title('Noise(Correntropy)');
            pause
        end
    end
end




