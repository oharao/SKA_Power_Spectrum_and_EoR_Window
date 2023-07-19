# SKA_Power_Spectrum_and_EoR_Window

Cambridge - HPC
-------
To run the Square Kilometer Array End-2-End Pipeline for Power Spectrum & EoR analysis the following steps may be taken to install the required dependencies, on the HPC. 

1. ``module load singularity``
2. ``singularity build --sandbox test docker://oscarohara/oskar_pipeline:v1.2``
3. Download the following data and locate in the relevant locations according to the 'Project File Tree':
    - The de-sourced GLEAM catalogue (``GLEAM_EGC_v2.fits``) may be obtained on Vizer [here](http://cdsarc.u-strasbg.fr/viz-bin/Cat?VIII/100#/browse), and should be located at ``sky_map/`` before being unzipped using ``gzip -d GLEAM_EGC_v2.fits.gz``
4. Modify the slurm submission script, an example of which may be found [here](https://github.com/oharao/SKA_Power_Spectrum_and_EoR_Window/blob/main/Documentation/slurm_ska_pipeline_example.txt)


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
├── sky_map
│   ├── GLEAM_EGC_v2.py
│   ├── read_gleam.py
│   └── gsm_gleam.py
├── End2End
│   ├── Coaxial_Transmission.py
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
└── t21_interpolation.py
```

Project Environment & Dependencies 
----------------------------------
To run the Square Kilometer Array End-2-End Pipeline for Power Spectrum & EoR analysis the following steps may be taken
to install the required dependencies, on an HPC service.
1. ``module load singularity``
2. ``singularity build --sandbox oskar_pipeline docker://oscarohara/oskar_pipeline``


#### Updating Docker Hub 
Once the Dockerfile has been modified, the image must be rebuilt before being pushed to Docker Hub for distribution. 
1. Once in the repository root directory the Image may be built: ``docker build . -t oscarohara/oskar_pipeline:v0.0`` 
whilst correctly enumerating the version number with respect to the original. 
2. The Image may then be pushed to Docker Hub for distribution to HPC services
``docker push oscarohara/oskar_pipeline:v0.0`` 
