#! /bin/bash

#################################### Bash colours ##########################################
NC='\e[m' # No Color
RED='\e[31m'
GREEN='\e[32m'
YELLOW='\e[33m'


################################## Set up error reporting ##################################
err_report() {
	echo -e "${RED}Installation failed. Error on line $1${NC}"
	exit 1
}

set_trap() {
	trap 'err_report $LINENO' ERR
}

unset_trap() {
	trap '' ERR
}

# Enable ERR trap handling
set_trap

####################################### Header line ########################################
echo "Movidius Neural Compute Toolkit Setup."

MAKE_PROCS=`nproc`
SUPPORTED_TF_VER=1.4.0

################################ Supported OS check ########################################
# XXX: Add all supported OSes check here. The rest should exit early.
echo "Checking OS and version..."

DISTRO="$(lsb_release -i 2>/dev/null | cut -f 2)"
VERSION=$(lsb_release -r 2>/dev/null | awk '{ print $2 }' | sed 's/[.]//')
OS_DISTRO="${DISTRO:-INVALID}"
OS_VERSION="${VERSION:-255}"
if [ "${OS_DISTRO,,}" == "ubuntu" ] || [ $OS_VERSION == 1604 ]; then
	echo "Installing on Ubuntu 16.04"
elif [ "${OS_DISTRO,,}" == "raspbian" ] && [ $OS_VERSION -ge 91 ]; then
	echo "Installing on Raspbian Stretch"
elif [ "${OS_DISTRO,,}" == "raspbian" ] && [ $OS_VERSION -ge 80 ] && [ $OS_VERSION -lt 90 ]; then
	ERR_MSG=$(cat <<- END
		${RED}Hmm, you are running Raspbian Jessie, which is not supported by NCSDK 1.09.
		Please upgrade to Raspbian Stretch and then proceed to install NCSDK.\n${NC}
		END
		)
	printf "$ERR_MSG"
	exit 1
else
	echo "Your current combination of linux distribution and distribution version is not officially supported!"
	exit 1
fi

################################# Read input config files ##################################
CONF_FILE=./ncsdk.conf
INSTALL_INFO_FILENAME=.ncsdk_install.info

# Read configration from file, and discard any malformed lines as this could
# be a malicious attack
function cfgfile_check_errs_fn
{
	count=0
	mal_count=0
	while IFS='' read -r line || [[ -n "$line" ]]; do
		count=$((count+1))
		if ! [[ $line =~ ^[^=]*+=[^=]*+$ ]]; then
			mal_count=$((mal_count+1))
			echo "Malformed line at line no. $count"
		fi
	done < $1

	echo $INSTALL_TENSORFLOW

	return 0;
}

# Check config file for errors
cfgfile_check_errs_fn $CONF_FILE

if [ "$mal_count" -gt "0" ]; then
	echo "Found $mal_count errors in config file. Please fix errors before running setup."
	exit 1
fi

# Config file error check done, source it
source $CONF_FILE

################################# Set verbosity level ######################################
APT_QUIET=-qq
DD_QUIET=status=none
PIP_QUIET=--quiet
GIT_QUIET=-q
STDOUT_QUIET='>/dev/null'
#STDERR_QUIET='2>/dev/null'
#STDOUTERR_QUIET='&>/dev/null'

if [ $VERBOSE == "yes" ]; then
	APT_QUIET=
	DD_QUIET=
	PIP_QUIET=
	GIT_QUIET=
	STDOUT_QUIET=
	STDERR_QUIET=
	STDOUTERR_QUIET=
fi

#TODO: What all is this needed for if only needed for apt, then is there a way to quit it
# after apt packages are all installed?
############################### Ask for SUDO permissions ##################################
# If the executing user is not root, ask for sudo priviledges
SUDO_PREFIX=""
PIP_PREFIX=""
if [ $EUID != 0 ]; then
	SUDO_PREFIX="sudo"
	sudo -v

        if [[ $SYSTEM_INSTALL == "yes" ]]; then
                PIP_PREFIX="$SUDO_PREFIX -H"
        fi

	# Keep-alive: update existing sudo time stamp if set, otherwise do nothing.
	while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &
fi

########################## Store the location of the script ###############################
# Find where this script is located. Store it in DIR
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
	DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
	SOURCE="$(readlink "$SOURCE")"
	[[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

########################## Log all actions in a log file ##################################
# Print stdout and stderr in screen and in logfile
d=$(date +%y-%m-%d-%H-%M)
LOGFILE="setup_"$d".log"
mkdir -p $DIR/setup-logs
exec &> >(tee -a "$DIR/setup-logs/$LOGFILE")

##################### Ask user for setup dir, else use default ############################
SETUPDIR_ORIG=$SETUPDIR
#read -e -p "Enter installation location (default: $SETUPDIR, press enter for default location): " SETUPDIR

#SETUPDIR=${SETUPDIR:-$SETUPDIR_ORIG}

#################### Check for previously installed versions and remove them ##############
function compare_versions
{
	VERCOMP_RETVAL=-1
	if [[ $1 == $2 ]]; then
		VERCOMP_RETVAL=0
	fi

	if [[ $1 == `echo -e "$1\n$2" | sort -V | head -n1` ]]; then
		VERCOMP_RETVAL=1
	else
		VERCOMP_RETVAL=2
	fi
}

INSTALLER_NCSDK_VER=`cat $DIR/version.txt`
echo "Installer NCSDK version: $INSTALLER_NCSDK_VER"

PREV_INSTALL_INFO=`$SUDO_PREFIX find /opt /home -name $INSTALL_INFO_FILENAME -print 2> /dev/null`
if [[ ! -z $PREV_INSTALL_INFO ]]; then
	PREV_OWNER=$(ls -l $PREV_INSTALL_INFO | awk '{print $3}')
	if [ $PREV_OWNER != $USER ]; then
		echo "Previous installation not owned by current user, continue at your own risk"
	fi
	PREV_INSTALL_DIR=${PREV_INSTALL_INFO%/*}
	PREV_NCSDK_VER=`cat $PREV_INSTALL_DIR/version.txt`
	echo "NCSDK version $PREV_NCSDK_VER previously installed at $PREV_INSTALL_DIR"

	compare_versions $PREV_NCSDK_VER $INSTALLER_NCSDK_VER

	#TODO: remove older version, or update as necessary
	if [[ $VERCOMP_RETVAL == 0 ]]; then
		echo "Previously installed version is the same as installer version, overwriting..."
	elif [[ $VERCOMP_RETVAL == 1 ]]; then
		echo "Previously installed version is older than installer version, upgrading..."
	else
		echo "Previously installed version is more recent than installer version, downgrading..."
	fi
fi

# Create setup directory
echo "Creating setup directory..."
mkdir -p "$SETUPDIR" &>/dev/null || $SUDO_PREFIX mkdir -p "$SETUPDIR"
$SUDO_PREFIX chown $(id -un):$(id -gn) "$SETUPDIR"

# Get absolute dir
SETUPDIR="$( cd "$SETUPDIR" && pwd )"


# Install toolkit and API in /usr/local/*
cd $DIR
SDK_DIR=$DIR/ncsdk-$(eval uname -m)

if [ -z "$INSTDIR" ]; then
    if [ -d /usr/local/lib ]; then
	INSTDIR=/usr/local
    else
	INSTDIR=/usr
    fi
fi

printf "Remove previous NCSDK references from PYTHONPATH..."
sed -i "\#export PYTHONPATH=\"\${PYTHONPATH}:"${SETUPDIR}"/mvnc/python\"#d" $HOME/.bashrc
sed -i "\#export PYTHONPATH=\$env:\""${SETUPDIR}"/mvnc/python\":\$PYTHONPATH#d" $HOME/.bashrc
printf "done\n"

# Create those directories if needed
$SUDO_PREFIX mkdir -p $INSTDIR/include/
$SUDO_PREFIX mkdir -p $INSTDIR/lib/
$SUDO_PREFIX mkdir -p $INSTDIR/lib/mvnc

################## Handle different OS distributions and versions #########################
function exit_on_error
{
	# Execute the incoming command, see if there were any errors. Exit on error.
	$1 |& tee file.txt
	fileout=$(<file.txt)
	echo "$fileout" | grep -q -e "Temporary failure resolving"
	if [ $? -eq 0 ]; then
		echo -e "${RED}Installation failed: Unable to reach remote servers, check your network and/or proxy settings and try again.${NC}"
		exit 128
	fi
}

# Ubuntu dependencies handler
function install_ubuntu_dependencies
{
	# Install apt dependencies
	exit_on_error "$SUDO_PREFIX apt-get $APT_QUIET update"
	exit_on_error "$SUDO_PREFIX apt-get $APT_QUIET install -y unzip coreutils curl git python3 python3-pip"
	exit_on_error "$SUDO_PREFIX apt-get $APT_QUIET install -y $(cat "$DIR/requirements_apt.txt")"
	exit_on_error "$SUDO_PREFIX apt-get $APT_QUIET install -y --no-install-recommends libboost-all-dev"
	$SUDO_PREFIX ldconfig
}

# Raspbian dependencies handler
# TODO: populate this correctly
function install_raspbian_dependencies
{
        exit_on_error "$SUDO_PREFIX apt-get $APT_QUIET update"
        exit_on_error "$SUDO_PREFIX apt-get $APT_QUIET install -y unzip coreutils curl git python3 python3-pip"
        exit_on_error "$SUDO_PREFIX apt-get $APT_QUIET install -y $(cat "$DIR/requirements_apt.txt")"
        exit_on_error "$SUDO_PREFIX apt-get $APT_QUIET install -y --no-install-recommends libboost-all-dev"
        $SUDO_PREFIX ldconfig
}

function find_tensorflow ()
{
	$PIP_PREFIX pip3 show $1 1> /dev/null
	if [ $? == 0 ]; then
		tf_ver_string=$($PIP_PREFIX pip3 show $1 | grep "^Version:")
		tf_ver=${tf_ver_string##* }
		flavour=""

		if echo "$1" | grep -q "gpu"; then
			hw="GPU"
		else
			hw="CPU"
		fi

		echo "Found tensorflow $hw version "$tf_ver"."
		if [ $tf_ver == "$SUPPORTED_TF_VER" ]; then
			return 0
		else
			return 1
		fi
	else
		return 1
	fi
}

function install_ubuntu_python_dependencies
{
        echo "Installing python dependencies..."

        # If sudo is used, pip packages will be installed in the systems directory, else
        # be installed per user, or in a virtual environment (if virtualenv or venv are$
        $PIP_PREFIX pip3 install $PIP_QUIET -r "$DIR/requirements.txt"
        $PIP_PREFIX pip install $PIP_QUIET "Enum34>=1.1.6"

	########################### Install tensorflow ###########################################
	if [ $INSTALL_TENSORFLOW == "yes" ]; then
		echo "Checking whether tensorflow is installed..."
		tf_ver_string=""
		tf_gpu_ver_string=""

		echo "looking for tensorflow CPU version..."
		find_tensorflow "tensorflow"
		tf=$?

		if [[ $tf -ne 0 ]]; then
			echo "looking for tensorflow GPU version..."
			find_tensorflow tensorflow-gpu
			tf_gpu=$?
		fi

		if [[ $tf -ne 0 && $tf_gpu -ne 0 ]]; then
			echo "Couldn't find a supported tensorflow version, installing tensorflow v$SUPPORTED_TF_VER"
			$PIP_PREFIX pip3 install $PIP_QUIET "tensorflow==$SUPPORTED_TF_VER"
		else
			echo "tensorflow already at latest supported version...skipping."
		fi
	fi
}

#todo: collapse apt entries into apt requirements
function install_raspbian_python_dependencies
{
        echo "Installing python dependencies..."

        # If sudo is used, pip packages will be installed in the systems directory, else
        # be installed per user, or in a virtual environment (if virtualenv or venv are$

	# for Raspian, we need to prefer apt with python3-* if available.

	exit_on_error "$PIP_PREFIX pip3 install $PIP_QUIET Cython"
	exit_on_error "$SUDO_PREFIX apt-get $APT_QUIET install -y python3-markdown"
	#Pillow already installed
	#PyYAML already installed
	exit_on_error "$PIP_PREFIX pip3 install $PIP_QUIET graphviz"
	exit_on_error "$PIP_PREFIX pip3 install $PIP_QUIET --upgrade numpy==1.13.1"
	exit_on_error "$SUDO_PREFIX apt-get $APT_QUIET install -y python3-h5py"
	exit_on_error "$SUDO_PREFIX apt-get $APT_QUIET install -y python3-lxml"
	exit_on_error "$SUDO_PREFIX apt-get $APT_QUIET install -y python3-matplotlib"
	exit_on_error "$SUDO_PREFIX apt-get $APT_QUIET install -y python3-protobuf" #warning, is only 3.0.0-9.  is this good enough???
	exit_on_error "$SUDO_PREFIX apt-get $APT_QUIET install -y python3-dateutil"
	exit_on_error "$PIP_PREFIX pip3 install $PIP_QUIET scikit-image"
	exit_on_error "$SUDO_PREFIX apt-get $APT_QUIET install -y python3-scipy"
	exit_on_error "$SUDO_PREFIX apt-get $APT_QUIET install -y python3-six"
	exit_on_error "$SUDO_PREFIX apt-get $APT_QUIET install -y python3-networkx"
}

function download_caffe()
{
	if [ -h "caffe" ]; then
		$SUDO_PREFIX rm caffe
	fi

	if [ ! -d $CAFFE_DIR ]; then
		echo "Downloading Caffe..."
		eval git clone $GIT_QUIET $CAFFE_SRC $STDOUT_QUIET $CAFFE_DIR
	fi

	ln -sf $CAFFE_DIR caffe
}

# Configure caffe to your heart's content
function configure_caffe_options
{
	# TODO: Make this configurable. Supports only python3 for now.
	sed -i 's/python_version "2"/python_version "3"/' CMakeLists.txt

	# Use/don't use CUDA
	if [ $CAFFE_USE_CUDA == "no" ]; then
		sed -i 's/CPU_ONLY  "Build Caffe without CUDA support" OFF/CPU_ONLY  "Build Caffe without CUDA support" ON/' CMakeLists.txt
	fi
}

function find_and_try_adjusting_symlinks()
{
	cd "$SETUPDIR"

	if [[ -h "caffe" && -d "$CAFFE_DIR" ]]; then
		readlink -f caffe | grep -q "$CAFFE_DIR"
		if [ $? -eq 0 ]; then
			echo "$CAFFE_DIR present and we're currently pointing to it"
		else
			if [ -d $CAFFE_DIR ]; then
				echo "$CAFFE_DIR present, but we're not currently using it. Use it."
			fi
			$SUDO_PREFIX rm -f caffe
			$SUDO_PREFIX ln -sf $CAFFE_DIR caffe
			return 0
		fi
	else
		echo "Possibly stale caffe install, starting afresh"
		return 1
	fi
}

function check_and_install_caffe()
{
	CAFFE_SRC="https://github.com/BVLC/caffe.git"
	CAFFE_VER="1.0"
	CAFFE_DIR=bvlc-caffe
	CAFFE_BRANCH=master
	if [ $CAFFE_FLAVOR == "intel" ] && [ "${OS_DISTRO,,}" == "ubuntu" ]; then
		CAFFE_SRC="https://github.com/intel/caffe.git"
		CAFFE_VER="1.0.3"
		CAFFE_DIR=intel-caffe
	elif [ $CAFFE_FLAVOR == "ssd" ]; then
		CAFFE_SRC="https://github.com/weiliu89/caffe.git"
		CAFFE_VER=""
		CAFFE_DIR=ssd-caffe
		CAFFE_BRANCH=ssd
	fi

	python3 -c "import caffe" 2> /dev/null
	if [ $? -eq 1 ]; then
		echo "Caffe not found, installing caffe..."
	else
		if [ $CAFFE_FLAVOR == "intel" ] && [ "${OS_DISTRO,,}" == "ubuntu" ]; then
			find_and_try_adjusting_symlinks

			# Look for intel caffe specific operation to determine the version of caffe currently running
			if [[ $? -eq 0 && -d "caffe" ]]; then
				cd caffe
				./build/tools/caffe time -model models/bvlc_googlenet/deploy.prototxt -engine "MKLDNN" -iterations 1 &> /dev/null
				if [ $? -eq 1 ]; then
					echo "Intel caffe requested but not found, installing Intel caffe..."
				else
					echo "Intel caffe already installed, skipping..."
					return 0
				fi
			fi
		else
			find_and_try_adjusting_symlinks
			if [ $? -eq 0 ]; then
				echo "Caffe already installed, skipping..."
				return 0
			fi
		fi
	fi

	############################## Install caffe ##############################################
	cd "$SETUPDIR"
	if [[ -h "caffe" && -d `readlink -f caffe` ]]; then
		echo `readlink -f caffe`
		cd `readlink -f caffe`
		# grep returns 1 if no lines are matched, causing the script to
		# think that installation failed, so append a "true"
		is_caffe_dir=`git remote show origin 2>&1 | grep -c -i $CAFFE_SRC` || true
		if [ "$is_caffe_dir" -eq 0 ]; then
			cd ..
			ERR_MSG=$(cat <<- END
				${YELLOW}The repo $SETUPDIR/caffe does not match the caffe project specified
				in this setup. Installing caffe from $CAFFE_SRC.\n${NC}
				END
				)
			printf "$ERR_MSG"

			download_caffe
			cd caffe
		fi

		eval git reset --hard HEAD $STDOUT_QUIET
		eval git checkout $GIT_QUIET $CAFFE_BRANCH $STDOUT_QUIET
		eval git branch -D fathom_branch &>/dev/null || true
		eval git pull $STDOUT_QUIET
	elif [ -d "caffe" ]; then
		ERR_MSG=$(cat <<- END
			${YELLOW}Found an old version of caffe, removing it to install the version
			specified in this installer from $CAFFE_SRC.\n${NC}
			END
			)
		printf "$ERR_MSG"

		$SUDO_PREFIX rm -r caffe
		download_caffe
	else
		download_caffe
	fi

	cd "$SETUPDIR"
	export PYTHONPATH=$env:"$SETUPDIR/caffe/python":$PYTHONPATH
	# At this point, caffe *must* be a symlink
	cd `readlink -f caffe`

	if [ "$CAFFE_BRANCH" != "master" ]; then
		eval git checkout $GIT_QUIET $CAFFE_BRANCH $STDOUT_QUIET
	fi

	if [ "$CAFFE_VER" != "" ]; then
		eval git checkout $GIT_QUIET $CAFFE_VER -b fathom_branch $STDOUT_QUIET
	fi

	configure_caffe_options

	echo "Compiling Caffe..."

	mkdir -p build
	cd build
	eval cmake .. $STDOUT_QUIET
	eval make -j $MAKE_PROCS all $STDOUT_QUIET

	echo "Installing caffe..."
	eval make install $STDOUT_QUIET
	# You can use 'make runtest' to test this stage manually :)

}

function check_and_remove_file()
{
	if [ -e "$INSTDIR/bin/$1" ]; then
		RECURSIVE=""
		if [ -d "$INSTDIR/bin/$1" ]; then
			RECURSIVE="-r"
		fi
		printf "Removing NCSDK toolkit file..."
		echo "$INSTDIR/bin/$1"
		$SUDO_PREFIX rm $RECURSIVE "$INSTDIR/bin/$1"
	fi
}



# There was a request to make the TK install optional, with the default being to install.
if [ $INSTALL_TOOLKIT == 'yes' ]; then
	############################## Install dependencies #######################################
	if [ "${OS_DISTRO,,}" == "ubuntu" ] || [ $OS_VERSION == 1604 ]; then
		install_ubuntu_dependencies
		install_ubuntu_python_dependencies
	elif [ "${OS_DISTRO,,}" == "raspbian" ] && [ $INSTALL_TENSORFLOW == "yes" ]; then
		ERR_MSG=$(cat <<- END
			${YELLOW}NOTE: Tensorflow is not officially supported on
			Raspbian Stretch, hence not installing this package.\n${NC}
			END
			)
		printf "$ERR_MSG"
		install_raspbian_dependencies
		install_raspbian_python_dependencies
	fi

	check_and_install_caffe

	# Add PYTHONPATH if not already there
	printf "Removing previous references to previous caffe installation..."
	# Remove some of the older varieties of references
	sed -i "\#export PYTHONPATH=\$env:\""${SETUPDIR}"/caffe/python\":\$PYTHONPATH#d" $HOME/.bashrc
	printf "done\n"
	echo "Adding caffe to PYTHONPATH"
	grep "^export PYTHONPATH=\"\${PYTHONPATH}:$SETUPDIR\/caffe\/python\"" $HOME/.bashrc || echo "export PYTHONPATH=\"\${PYTHONPATH}:$SETUPDIR/caffe/python\"" >> $HOME/.bashrc


	#################################### Install toolkit #####################################
	# Copy TK to destination
	# If installed, remove toolkit binaries
	check_and_remove_file mvNCCheck
	check_and_remove_file mvNCProfile
	check_and_remove_file mvNCCompile
	check_and_remove_file ncsdk

	$SUDO_PREFIX cp -r $SDK_DIR/tk $INSTDIR/bin/ncsdk
	$SUDO_PREFIX ln -s $INSTDIR/bin/ncsdk/mvNCCompile.py $INSTDIR/bin/mvNCCompile
	$SUDO_PREFIX ln -s $INSTDIR/bin/ncsdk/mvNCCheck.py $INSTDIR/bin/mvNCCheck
	$SUDO_PREFIX ln -s $INSTDIR/bin/ncsdk/mvNCProfile.py $INSTDIR/bin/mvNCProfile

fi # End optional TK install

# Copy uninstall script to destination
$SUDO_PREFIX cp $DIR/uninstall-ncsdk.sh $SETUPDIR/

# Copy FW to destination
$SUDO_PREFIX cp $SDK_DIR/fw/MvNCAPI.mvcmd $INSTDIR/lib/mvnc/

# Remove old debian packages from the 1.07.06 release
# Unset trap for ERR handling for this block alone, since a non-zero return code
# from dpkg queries result in early aborting of the script
unset_trap
function remove_ncapi_deb
{
	# Clean up old install artifacts
	old_python_mvnc=`dpkg -s $1`
	if [ $? -eq 0 ]; then
		printf "Removing old installed $1\n"
		$SUDO_PREFIX dpkg --purge $1
	fi
}

# IMPORTANT: Do not change the order
remove_ncapi_deb python3-mvnc
remove_ncapi_deb mvnc-dev
remove_ncapi_deb mvnc

# Set the trap handler again
set_trap

# Copy C API to destination
$SUDO_PREFIX cp $SDK_DIR/api/c/mvnc.h $INSTDIR/include/
$SUDO_PREFIX cp $SDK_DIR/api/c/mvnc_deprecated.h $INSTDIR/include/
$SUDO_PREFIX cp $SDK_DIR/api/c/libmvnc.so.0 $INSTDIR/lib/mvnc/
$SUDO_PREFIX rm -f $INSTDIR/lib/libmvnc.so
$SUDO_PREFIX rm -f $INSTDIR/lib/libmvnc.so.0

$SUDO_PREFIX ln -s $INSTDIR/lib/mvnc/libmvnc.so.0 $INSTDIR/lib/libmvnc.so.0
$SUDO_PREFIX ln -s $INSTDIR/lib/mvnc/libmvnc.so.0 $INSTDIR/lib/libmvnc.so

$SUDO_PREFIX ldconfig

# Copy other collaterals to destination
$SUDO_PREFIX cp -r $DIR/version.txt $SETUPDIR/
$SUDO_PREFIX cp -r $SDK_DIR/LICENSE $SETUPDIR/

# Install python API
$PIP_PREFIX pip3 install $PIP_QUIET --upgrade --force-reinstall $SDK_DIR/api
$PIP_PREFIX pip install $PIP_QUIET --upgrade --force-reinstall $SDK_DIR/api

echo "NCS Libraries have been installed in $INSTDIR/lib"
if [ $INSTALL_TOOLKIT = 'yes' ]; then
	echo "NCS Toolkit binaries have been installed in $INSTDIR/bin"
fi

echo "NCS Include files have been installed in $INSTDIR/include"
echo "NCS Python API has been installed in $SETUPDIR, and PYTHONPATH environment variable updated"

INSTALL_INFO_FILE=$SETUPDIR/$INSTALL_INFO_FILENAME
touch $INSTALL_INFO_FILE
echo "ncsdk_path=$SETUPDIR" > $INSTALL_INFO_FILE
echo "ncs_lib_path=$INSTDIR/lib" >> $INSTALL_INFO_FILE
echo "ncs_inc_path=$INSTDIR/include" >> $INSTALL_INFO_FILE

if [ $INSTALL_TOOLKIT = 'yes' ]; then
	echo "ncs_bin_path=$INSTDIR/bin" >> $INSTALL_INFO_FILE
fi

# Update udev rules
echo "Updating udev rules..."
$SUDO_PREFIX cp $SDK_DIR/udev/97-usbboot.rules /etc/udev/rules.d/
$SUDO_PREFIX udevadm control --reload-rules
$SUDO_PREFIX udevadm trigger

################### Final touch up ###############################
echo "Adding user '$USER' to 'users' group"
$SUDO_PREFIX usermod -a -G users $USER

echo -e "${GREEN}Setup is complete.${NC}
The PYTHONPATH enviroment variable was added to your .bashrc as described in the Caffe documentation. 
${YELLOW}Keep in mind that only newly spawned terminals can see this variable!
This means that you need to open a new terminal in order to be able to use the NCSDK.${NC}"

echo "Please provide feedback in our support forum if you encountered difficulties."

