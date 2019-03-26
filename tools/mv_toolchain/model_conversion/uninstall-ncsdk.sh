#! /bin/bash

INSTALL_INFO_FILENAME=.ncsdk_install.info

################################## Set up error reporting ##################################
err_report() {
    echo -e "${RED}Installation failed. Error on line $1${NC}"
}

trap 'err_report $LINENO' ERR

############################### Ask for SUDO permissions ##################################
# If the executing user is not root, ask for sudo priviledges
SUDO_PREFIX=""
if [ $EUID != 0 ]; then
	SUDO_PREFIX="sudo"
	sudo -v

	# Keep-alive: update existing sudo time stamp if set, otherwise do nothing.
	#while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &
fi

PREV_INSTALL_INFO=`$SUDO_PREFIX find /opt /home -name $INSTALL_INFO_FILENAME -print 2> /dev/null`
if [[ ! -z $PREV_INSTALL_INFO ]]; then
	PREV_OWNER=$(ls -l $PREV_INSTALL_INFO | awk '{print $3}')
	if [ $PREV_OWNER != $USER ]; then
		echo "Previous installation not owned by current user, continue at your own risk"
	fi
	PREV_INSTALL_DIR=${PREV_INSTALL_INFO%/*}
	PREV_NCSDK_VER=`cat $PREV_INSTALL_DIR/version.txt`
	echo "NCSDK version $PREV_NCSDK_VER previously installed at $PREV_INSTALL_DIR"
else
	echo "No valid install of NCSDK found, exiting"
	exit 0
fi

NCSDK_PATH=`grep 'ncsdk_path' $PREV_INSTALL_INFO | cut -d '=' -f 2`
NCS_BIN_PATH=`grep 'ncs_bin_path' $PREV_INSTALL_INFO | cut -d '=' -f 2`
NCS_LIB_PATH=`grep 'ncs_lib_path' $PREV_INSTALL_INFO | cut -d '=' -f 2`
NCS_INC_PATH=`grep 'ncs_inc_path' $PREV_INSTALL_INFO | cut -d '=' -f 2`
echo "NCSDK files are at: $NCSDK_PATH"
if [ ! -z $NCS_BIN_PATH ]; then
	echo "NCSDK binaries are at: $NCS_BIN_PATH"
fi
echo "NCSDK libraries are at: $NCS_LIB_PATH"
echo "NCSDK include files are at: $NCS_INC_PATH"

function check_and_remove_file
{
	if [ -e "$NCS_BIN_PATH/$1" ]; then
		RECURSIVE=""
		if [ -d "$NCS_BIN_PATH/$1" ]; then
			RECURSIVE="-r"
		fi
		printf "Removing NCSDK toolkit file..."
		echo "$NCS_BIN_PATH/$1"
		$SUDO_PREFIX rm $RECURSIVE "$NCS_BIN_PATH/$1"
	fi
}

# If installed, remove toolkit binaries
check_and_remove_file mvNCCheck
check_and_remove_file mvNCProfile
check_and_remove_file mvNCCompile
check_and_remove_file ncsdk

# Remove libraries
printf "Removing NCSDK libraries..."
$SUDO_PREFIX rm $NCS_LIB_PATH/libmvnc.so.0
$SUDO_PREFIX rm $NCS_LIB_PATH/libmvnc.so
$SUDO_PREFIX rm -r $NCS_LIB_PATH/mvnc
printf "done\n"

# Remove API, don't prompt user for a response
printf "Removing NCS python API\n"
$SUDO_PREFIX -H pip3 uninstall -y mvnc
$SUDO_PREFIX -H pip uninstall -y mvnc

# Remove include files
printf "Removing NCSDK include files..."
$SUDO_PREFIX rm $NCS_INC_PATH/mvnc.h
$SUDO_PREFIX rm $NCS_INC_PATH/mvnc_deprecated.h
printf "done\n"

# Remove udev rules files
printf "Removing udev rules..."
$SUDO_PREFIX rm /etc/udev/rules.d/97-usbboot.rules
printf "done\n"

# Remove NCSDK files
printf "Removing NCSDK files..."
$SUDO_PREFIX rm -r $NCSDK_PATH
printf "done\n"

printf "Running ldconfig..."
$SUDO_PREFIX ldconfig
printf "done\n"

printf "Updating udev rules..."
$SUDO_PREFIX udevadm control --reload-rules
$SUDO_PREFIX udevadm trigger
printf "done\n"

printf "Remove NCSDK references from PYTHONPATH..."
#sed -i "\#export PYTHONPATH=\"\${PYTHONPATH}:"${NCSDK_PATH}"/mvnc/python\"#d" $HOME/.bashrc
sed -i "\#export PYTHONPATH=\"\${PYTHONPATH}:"${NCSDK_PATH}"/caffe/python\"#d" $HOME/.bashrc
# Remove some of the older versions of this as well
sed -i "\#export PYTHONPATH=\$env:\""${NCSDK_PATH}"/mvnc/python\":\$PYTHONPATH#d" $HOME/.bashrc
sed -i "\#export PYTHONPATH=\$env:\""${NCSDK_PATH}"/caffe/python\":\$PYTHONPATH#d" $HOME/.bashrc
printf "done\n"

echo "Successfully uninstalled NCSDK from the system"

