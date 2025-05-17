export CUR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
conda activate env_isaaclab
export PYTHONPATH=$PYTHONPATH:$CUR/extensions/pegasus.simulator
