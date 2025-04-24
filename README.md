# Absolute-impedance-inversion-using-domain-adaptation
This repository contains the code for absolute impedance inversion using doamin adversarial neural network (DANN). 
Here, I have showed the implementation of DANN framework on the synthetic example. For this benchmark Marmousi2 model is used as a source domain (with sufficient labels) and SEAM subsalt earth model as target domain (very limited labels).
Considering the real data scenario, only 10% data of SEAM model is used for training.
Given the different underlying lithology of both datasets, i.e., Marmousi2 and SEAM, our implementation has shown a significant efficiency in recovering the complex salt features of the SEAM model.
Further, to recover the absolute impedance, we showed the efficacy of the classic envelope attribute to recover the missing low-frequency information from the band-limited seismic data. Along with instantaneous phase, to compensate for the lost phase information, while tacking the envelope. Finally, envelope and instantaneous phase attributes are fed as input along with the bandlimited seismic. This work we have showed in "Broadband acoustic impedance inversion using seismic multi-attributes and sequentialconvolution neural network". doi: https://doi.org/10.3997/2214-4609.202410820.
The implmentation of the DANN along with multi-attribute approach is discussed in "Seismic absolute acoustic impedance inversion using domain adversarial based transfer learning", doi: https://doi.org/10.1190/image2024-4099826.1

If you face 
