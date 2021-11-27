% Demonstration of using the SIDD Benchmark

% -------------------------------------------------------------------------
% Author: Abdelrahman Abdelhamed, York University
% kamel@eecs.yorku.ca
% https://www.eecs.yorku.ca/~kamel/
% -------------------------------------------------------------------------
% Reference: Abdelrahman Abdelhamed, Lin S., Brown M. S. 
% "A High-Quality Denoising Dataset for Smartphone Cameras", 
% IEEE Computer Vision and Pattern Recognition (CVPR'18), June 2018.
% -------------------------------------------------------------------------

% the directory containing the SIDD Benchmark images
% Note: change this line if the Data is not in the parent directory
% SiddDataDir = fullfile('..', 'Data');
SiddDataDir = fullfile('/home/clement/Documents/light_code/sidd_eval/SIDD_Benchmark_Data');
% SiddDataDir = fullfile('\home\clement\Documents\light_code\sidd_eval\SIDD_Benchmark_Data');

% path for a Gaussian noise estimatoin method
% Note: you may replace this smehtod with your favourite one
addpath 'NoiseEstimation_ICCV2015_code';

% Some optional, yet useful, data to include, feel free to put your data
OptionalData.MethodName = 'DummyDenoiser';
OptionalData.Authors = 'Jane Doe and Jone Doa';
OptionalData.PaperTitle = 'Dummy Image Denoising';
OptionalData.Venue = 'SIDD Demo';
% Specs of the machine used to run the benchmark (useful for time
% comparison)
OptionalData.MachineSpecs = 'Intel Core i7 6700 @ 3.4 GHz, 32 GB RAM'; 
% Note: you may add more optional data as needed

% denoise Raw-RGB images
% SIDD_Denoise(@DummyDenoiserRaw, SiddDataDir, 'raw', OptionalData);

% denoise sRGB images
SIDD_Denoise(@DummyDenoiserSrgb, SiddDataDir, 'srgb', OptionalData);

