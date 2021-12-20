# NEXUS_RF
`NEXUS_RF/DeviceControl/GPU_SDR` is the repository `nasa/GPU_SDR` available at https://github.com/nasa/GPU_SDR

`NEXUS_RF/DeviceControl/uhd` is the repository `EttusResearch/uhd` available at https://github.com/EttusResearch/uhd

When cloning the directory for the first time, do the following:
```
git clone git@github.com:nkurinsky/NEXUS_RF.git
cd NEXUS_RF
git submodule init
git submodule sync 
git submodule update --init --recursive
```
which should populate both of the git `submodule`s.

When updating the repository and you want updates in the submodules, run
```
git pull
git submodule update
```
