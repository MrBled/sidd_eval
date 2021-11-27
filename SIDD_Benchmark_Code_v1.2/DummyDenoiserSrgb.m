function [denoisedImage] = DummyDenoiserSrgb(noisyImage, noiseSigma)

denoisedImage = imgaussfilt(noisyImage);

end

