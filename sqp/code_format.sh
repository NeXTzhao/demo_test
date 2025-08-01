#!/bin/bash

set -e

bash_name=$(basename $0)
script_dir=$(cd $(dirname $0); pwd)
root_dir=$(cd $script_dir/..; pwd)

cd $root_dir

find mim_solvers -type f -regex '.*\.\(cpp\|hpp\|cc\|c\|h\)' -exec clang-format -style=file -i {} +
# black --check .