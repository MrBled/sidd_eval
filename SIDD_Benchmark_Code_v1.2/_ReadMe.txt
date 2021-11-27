*** Smartphone Image Denoising Dataset (SIDD) Benchmark
=======================================================

The "Code_v1.1" directory contains matlab code to evaluate denoising methods on the SIDD benchmark.

** To demonstrate the code: run Demo_SIDD.m

** To evaluate your denoising method:

1) Replace DummyDenoiserRaw.m and/or DummyDenoiserSrgb.m with your own functions.

2) In Demo_SIDD.m, change @DummyDenoiserRaw and/or @DummyDenoiserSrgb with your denoising function handles.

3) If you want to use some Gaussian noise estimatoin method, do not forget to call it in DenoiseSrgb.m.

4) In Demo_SIDD.m, you may fill in your OptionalData (e.g., method name, authors, paper title, specs of the machine used in benchmark, etc.).

5) After denoising is done, find your denoising results in the "<SiddDataDir>/../Submit" directory: "SubmitRaw.mat" and/or "SubmitSrgb.mat".

6) Submit your results (files "SubmitRaw.mat" and/or "SubmitSrgb.mat") through the benchmark page: https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php
Your results will be evaluated and sent back to the email address you specify, with an optoin to post your results on the benchmark website.

Feel free to contact me, Abdelrahman, at kamel@eecs.yorku.ca .

Enjoy!