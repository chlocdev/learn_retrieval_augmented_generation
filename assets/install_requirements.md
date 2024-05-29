# install requirements python packages

- to install python packages: `pip install -r requirements.tx`

# install nvidia-driver

- update: `sudo apt update`

- check driver version: `ubuntu-drivers devices`

- install NVIDIA driver `sudo apt install nvidia-driver-xxx` (GeForce RTX 3060, xxx = 535)

- reboot `sudo reboot`

- check `nvidia-smi`

- check cuda available ` python3 -c "import torch ; print(torch.cuda.is_available())" `