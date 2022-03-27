# ROB 530 Project Development Docker Container

Container that bundles ROS 1's Noetic release, with the full desktop install and all dependencies needed for development.

This setup assumes:

1. A *Nix like environment (Linux, WSL2 if using Windows, or Mac (untested))
1. Docker is installed and configured. See [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/). On Windows, ensure that you've enabled the WSL integration for Docker (Settings->Resources->WSL Integration)

## Steps to use:

1. Edit `./.env` and modify any variables
1. Run `./launch.sh` to start the docker container. You should find yourself in a docker container with ROS Noetic installed. Run `env | grep ROS` to make sure everything is configured properly
1. If you want to attach to a running `rob530_dev` container, simply run `./attach.sh` from another terminal
