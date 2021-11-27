function [denoisedImage] = DummyDenoiserRaw(noisyImage, NLF)

denoisedImage = imgaussfilt(noisyImage);

end

