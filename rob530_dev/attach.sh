#! /bin/bash

# Copyright (c) 2019 Javier Peralta Saenz, Ariel Mora Jimenez.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Ryan D. Lewis

# Script Configuration
source .env

sudo docker exec -it \
	--user=$USER_ID \
	$CONTAINER_NAME bash -c "source /opt/ros/foxy/setup.bash && bash"\
