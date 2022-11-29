# SKA_Power_Spectrum_and_EoR_Window

Cambridge - HPC
-------
To run the Square Kilometer Array End-2-End Pipeline for Power Spectrum & EoR analysis the following steps may be taken to install the required dependencies, on the HPC. 

1. ``module load singularity``
2. ``singularity build --sandbox test docker://oscarohara/oskar_pipeline``
3. Download the following data and locate in the relevent locations according to the 'Project File Tree':
    - ``/End2End/GLEAM_EGC.fits`` [here](https://drive.google.com/file/d/15oMSprZ0NFO_ttAN6pDsPX--0jn9XJH1/view?usp=sharing)
    - ``70-100MHz_control.vis`` (control visibilities) [here](https://drive.google.com/drive/folders/10JoGY3ugB64NC7LbG3OIxfjRUSkdrRZB?usp=sharing)
    - ``/comoving/70-100MHz`` (21cm signal) [here](https://drive.google.com/drive/folders/1NSy2XSJM4vR3RtV1ku1Gf5xSz7CCSngW?usp=sharing)
5. Modify the slurm submission script, an example of which may be found [here]()


Project File Tree
------------------

```
SKA_Power_Spectrum_and_EoR_Window
├── Cable Decay .ipynb
├── Dockerfile
├── Documentation
│   └── images
│       ├── coax_structure.png
│       └── project_layout.png
├── End2End
│   ├── Coaxial_Transmission.py
│   ├── GLEAM_EGC.fits
│   ├── OSKAR_default_script.py
│   ├── __pycache__
│   │   └── generate_EoR.cpython-38.pyc
│   ├── antenna_pos
│   │   ├── layout_wgs84.txt
│   │   ├── position.txt
│   │   ├── station000
│   │   │   └── layout.txt
│   │   ├── station001
│   │   │   └── layout.txt
│   ├── antenna_pos_core
│   │   ├── layout_wgs84.txt
│   │   ├── position.txt
│   │   ├── station000
│   │   │   └── layout.txt
│   │   ├── station001
│   │   │   └── layout.txt
│   ├── generate_EoR.py
│   ├── image_ps.py
│   ├── logger.py
│   ├── run.py
│   └── test.py
├── Image_plot.py
├── README.md
├── comoving
│   └── 70-100MHz
│       ├── delta_los_comoving_distance.csv
│       ├── freq_70.000_MHz_interpolate_T21_slices.fits
│       ├── freq_70.012_MHz_interpolate_T21_slices.fits
│       ├── freq_70.024_MHz_interpolate_T21_slices.fits
│       ├── los_comoving_distance.csv
│       └── pixel_size_deg.csv
└── t21_interpolation.py
70-100MHz_control.vis
├── gleam_all_freq_70.000_MHz.vis
├── gleam_all_freq_70.012_MHz.vis
├── gleam_all_freq_70.024_MHz.vis
```

