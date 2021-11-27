function [DenoisedBlocksRaw, TimeMP] = ...
    DenoiseRaw(Denoiser, siddDataDir)

% DENOISERAW Deanoises SIDD benchmark images in raw-RGB space using some
% Denoiser.

% Denoiser: a function handle to the denoiser to be evaluated on the SIDD
% Benchmark

% DenoisedBlocksRaw: the denoised blocks from raw-RGB images.

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
tmp = load(fullfile(siddDataDir, '..', 'BenchmarkBlocks32.mat')); 
BenchmarkBlocks32 = tmp.BenchmarkBlocks32;
nBlocks = size(BenchmarkBlocks32, 1);

DenoisedBlocksRaw = cell(nImages, nBlocks);
TimeMP = 0; % denoising time (in seconds) per megapixel
bi=0;
% for each image
for i = 1 : nImages
    
    % load noisy raw-RGB image i
%     tmp = load(fullfile(siddDataDir, imageFiles(i).name, 'NOISY_RAW_010.MAT'));
    [~, sfn, ext] = fileparts(imageFiles(i).name);
    ss = strsplit(sfn, '_');
    tmp = load(fullfile(siddDataDir, imageFiles(i).name, [char(ss(1)),'_NOISY_RAW_010.MAT']));
    
    noisyImage = tmp.x;
    
    %imshow(noisyImage.*20);
%     pause(1);
    
    % load metadata of image i
%     tmp = load(fullfile(siddDataDir, imageFiles(i).name, 'METADATA_RAW_010.MAT'));
    [~, sfn, ext] = fileparts(imageFiles(i).name);
    ss = strsplit(sfn, '_');
    tmp = load(fullfile(siddDataDir, imageFiles(i).name, [char(ss(1)),'_METADATA_RAW_010.MAT']));
    
    metadata = tmp.metadata;
    
    % noise level function of image i
    NLF = GetNLF(metadata);
    
    % for each block
    for b = 1 : nBlocks
        
        fprintf('Denoising raw-RGB image %02d, block %02d ... ', i, b);
        
        bi = BenchmarkBlocks32(b, :);
        noisyBlock = noisyImage(bi(1) : bi(1) + bi(3) - 1, bi(2) : bi(2) + bi(4) - 1);
        
        % denoise CFA channels separately
        blockTime = 0;
        denoisedBlock = zeros(size(noisyBlock));
        for u = 1 : 2
            for v = 1 : 2
                t0 = tic; % start timer
                denoisedBlock(u : 2 : end, v : 2 : end) = Denoiser(noisyBlock(u : 2 : end, v : 2 : end), NLF);
                t1 = toc(t0); % stop timer
                blockTime = blockTime + t1;
                DenoisedBlocksRaw{i, b} = single(denoisedBlock);
            end
        end
        
        % total time
        TimeMP = TimeMP + blockTime;
        fprintf('Time = %f seconds\n', blockTime);
    end
end

TimeMP = TimeMP * 1024 * 1024 / ...
    (nImages * nBlocks * bi(3) * bi(4)); % seconds per megapixel

end