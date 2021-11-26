# SKA_Power_Spectrum_and_EoR_Window

Windows
-------
To run the SKA Power Spectrum & EoR Window scripts on Windows we must first install the required dependencies, this primarily involves WSL, Ubuntu and Casa. 

You will need a x86 PC running Windows 10.

Windows 10/11 needs to be updated to include the Windows 10 Fall Creators update, released October 2017. This update includes the Windows Subsystem for Linux which is needed to run the Ubuntu terminal.

1. To enable WSL the following commands must be ran in PowerShell as an Administrator:
  - WSL 1 for Windows 10 Fall Creators update and newer: ``dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart``

  - WSL 2 for Windows 10 May 2020 update and newer: ``dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart``
  
2. Restart your local machine. 

3. Ubuntu may then be installed from the microsoft store [here](https://www.microsoft.com/en-us/p/ubuntu/9nblggh4msv6). Launch Ubuntu and configure the local systems username and password. 

4. Download the relevant .tar file [here](https://casa.nrao.edu/casa_obtaining.shtml) and place it in a work directory (e.g. ~/casa).

5. From a Linux terminal window, expand the casa-xyz.tar.gz (CASA 6.1)  or casa-release-xyz.tar.gz (CASA 5.7) file: ``$ tar -xvf casa-xyz.tar.xz``.

6. The scripts may now be run using the following command: ``~$ casa/casa-6.4.0-16/bin/casa -c ./OSKAR_data/yqyp2_partiii/proj_part2_old.py``


Required File Tree
------------------

```
user/
├─ casa/
│  ├─ casa-6.4.0-16/
│  │  ├─ bin/
│  │  │  ├─ casa
├─ SKA_EoR_windows/
│  ├─ proj_part1.py
│  ├─ proj_part1_v2.py
│  ├─ proj_part2_old.py
```

