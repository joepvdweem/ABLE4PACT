# Photoacoustic Image Reconstruction Using Multilayer Neural Networks
## Abstract
Photoacousic Tomography (PACT) is an emerging
field in Biomedical Imaging used for e.g. breast cancer detection
and brain lesion detection. Current conventional reconstruction
techniques use Delay and Sum (DAS) based backprojection
algorithms to reconstruct images, however these suffer from
PACT’s practical limitations. The main limitations are: Limited bandwidth, limited view, lossy medium and hetergeneous
medium. This paper proposes an end-to-end deep learning approach for the reconstruction of PACT images constisting of two
seperate neural networks. The first network suppresses sidelobes
caused by beamforming using a beamforming network (BFN)
tackling the limited view problem. It also extends the bandwidth
of the bandlimited data using a bandwidth extension network
(BWN). These are trained using and analysed using four training
strategies to evaluate the performamnce. First proposed BFNs
are trained using capon beamformed in vivo data and evaluated.
Secondly the BWN is tested using DAS images based on simulated
data and as training target the original intensity map. The third
is training the entire network on only simulated data and lastly
the network is trained on both simulated and enhanced in vivo
data. From the results of these networks it can be concluded
that the BWN network proves powerful in finding the underlying
intensity map. The combined use with the beamforming network
shows good results in enhancing the contrast near the center
of the image. In the results it can be observed that the main
limiting factor right now is the beamforming network as it does
not enhance the negative part of the signal. It does, however,
show great results in enhancing the resolution of the image.
