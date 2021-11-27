function [DenoisedBlocksSrgb, TimeMP] = ...
    DenoiseSrgb(Denoiser, siddDataDir)

% DENOISESRGB Deanoises SIDD benchmark images in sRGB space using some
% Denoiser.

% Denoiser: a function handle to the denoiser to be evaluated on the SIDD
% Benchmark

% DenoisedBlocksSrgb: the denoised blocks from sRGB images.

% siddDataDir: the directory containing the SIDD Benchmark images

% -------------------------------------------------------------------------
% Author: Abdelrahman Abdelhamed, York University
% kamel@eecs.yorku.ca
% https://www.eecs.yorku.ca/~kamel/
% -------------------------------------------------------------------------
% Reference: Abdelrahman Abdelhamed, Lin S., Brown M. S. 
% "A High-Quality Denoising Dataset for Smartphone Cameras", 
% IEEE Computer Vision and Pattern Recognition (CVPR'18), June 2018.
% -------------------------------------------------------------------------

% list of all image directories (there should be 40 of them)
imageFiles = dir(siddDataDir);
imageFiles = imageFiles(3:end);
%nImages = numel(imageFiles);
nImages = 40;

% block positions
tmp = load(fullfile(siddDataDir, 'BenchmarkBlocks32.mat')); 
BenchmarkBlocks32 = tmp.BenchmarkBlocks32;
nBlocks = size(BenchmarkBlocks32, 1);

DenoisedBlocksSrgb = cell(nImages, nBlocks);
TimeMP = 0; % denoising time (in seconds) per megapixel

% for each image
for i = 1 : nImages
    
    % load noisy raw-RGB image i
%     noisyImage = imread(fullfile(siddDataDir, imageFiles(i).name, ...
%         'NOISY_SRGB_010.PNG'));
%     noisyImage = im2double(noisyImage); 
    [~, sfn, ext] = fileparts(imageFiles(i).name);
    ss = strsplit(sfn, '_');
    noisyImage = imread(fullfile(siddDataDir, imageFiles(i).name, ...
        [char(ss(1)),'_NOISY_SRGB_010.PNG']));
    noisyImage = im2double(noisyImage);
    
    %imshow(noisyImage);
    
    % estimate of noise Sigma for image i
    % Note: you may replace this with your favourite estimation method
    fprintf('Estimating noise Sigma for image %02d ... \n', i);
    %%%noiseSigma = NoiseEstimation(noisyImage, 8);
    noiseSigma = []; % delete this line to use noiseSigma
    
    % for each block
    for b = 1 : nBlocks
        
        fprintf('Denoising sRGB image %02d, block %02d ... ', i, b);
        
        bi = BenchmarkBlocks32(b, :);
        noisyBlock = noisyImage(...
            bi(1) : bi(1) + bi(3) - 1, ...
            bi(2) : bi(2) + bi(4) - 1, :);
        
        % denoise 3 RGB channels simultaneously
        t0 = tic; % start timer
        denoisedBlock = Denoiser(noisyBlock, noiseSigma);
        t1 = toc(t0); % stop timer
        DenoisedBlocksSrgb{i, b} = im2uint8(denoisedBlock);
        
        % total time
        TimeMP = TimeMP + t1;
        fprintf('Time = %f seconds\n', t1);
    end
end

TimeMP = TimeMP * 1024 * 1024 / ...
    (nImages * nBlocks * bi(3) * bi(4)); % seconds per megapixel

end