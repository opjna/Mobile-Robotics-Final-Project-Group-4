ARG BASE_IMAGE=osrf/ros:noetic-desktop

# Install dev packages
FROM $BASE_IMAGE as base
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
		git \
		tmux \
		curl \
		wget \
		vim \
		sudo \
		unzip \
		python3-pip \
		&& apt-get clean \
		&& rm -rf /var/lib/apt/lists/*

# Install ROS Packages
RUN apt-get update && apt-get install -y \
		&& apt-get clean \
		&& rm -rf /var/lib/apt/lists/*

# Install pip3 packages
RUN pip3 install numpy scipy pyyaml matplotlib sympy tk

CMD ["bash"]
