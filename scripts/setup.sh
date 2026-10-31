#!/usr/bin/env bash
# One-shot setup: venv + dependencies + .env
# Usage:  bash scripts/setup.sh          (light install: Module 1 + tests)
#         bash scripts/setup.sh --all    (everything, incl. torch/chroma/gradio/mcp)


#**********************************************************************************************************
# shebang line (also known as a hashbang or interpreter directive):
#   Placed at the absolute first line of a script file, 
#   it tells the Linux/Unix operating system which exact program to use to 
#   read and execute the commands written in that script file.
#   Combine # (Hash/Sharp) with ! (Bang) to #! that acts as a "magic number" signature for Linux kernel. 
#
# e.g.,
#  #!/usr/bin/bash:     Looks at the exact path /usr/bin/bash for the command to run this script.
#  #!/usr/bin/env bash: Uses the /usr/bin/env to search enviromental variable $PATH for the bash binary.
#  #!/usr/bin/python3:  a shebang line for Python to use /usr/bin/python3 to run this script.


#**********************************************************************************************************
# set is a bash built-in command.  
# 
# set -euo pipefail (or set -e -u -o pipefail) is known as the foundation of "Bash Strict Mode".
# It is a combination of three distinct flags designed to make Bash scripts safer for bash built-in command
# set -e: errexit, exits immediately if any command returns a non-zero exit status (a failure).
# set -u: nounset, treats unset/undefined variables as an error and exits immediately. 
#                  catches typos in variable names 
#                  e.g., executing rm -rf $BACP_DIR/* because you misspelled BACKUP_DIR.
# set -o pipefail: sets the exit code of a pipeline to the last command that failed, 
#                  rather than the final command in the chain.
#                  e.g., cat missiong_file.txt|grep keyword
#                  cat fails but grep succeeds with returning nothing. 
#                  Normally the exit status $? shows 0 (the grep exit status)
#                  If set -o pipefail, $? shows 1 (the exit status of failed command cat in the pipe) 

set -euo pipefail

#*********************************************************************************************************
# Bash build-in environment variables:
# 
# $PATH:      The environmental variable that tells the shell which directories to search when you run a command.
#             e.g., echo $PATH outpus: /usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
#
# $SHELLOPTS: A read-only, colon-separated bash built-in environment variable.
#             It lists all currently enabled shell options controlled by the set command
#             on "set -euo pipefile", $SHELLOPTS shows: errexit:nounset:pipefail
#


#*********************************************************************************************************
# Built-in Special Positional & Status Variables:
#
# $?: represents the exit status (or return code) of the immediately preceding command. 
#     It is an integer ranging from 0 to 255, 0 for Success and 1 to 255 for different failures.
#
# $0: the special positional shell variable that holds the fullpath name of the running shell script
# $1: the special positional shell variable that holds first parameter of the running shell script
# $2: the special positional shell variable that holds second parameter of the running shell script
# $#: the special shell variable that holds The number of arguments passed to the runnng shell script



#*********************************************************************************************************
# Command substitution:
#
# $(command): equivalent to legacy syntax in backticks: `command`
#             this is a shell feature that allows you to run a shell command 
#             and capture its standard output directly into a variable 
#             e.g., current_time=$(date) or current_time=`date`



#*********************************************************************************************************
# dirname:      the command extracts the directory portion from a given file path
# dirname "$0": in this script, outputs: /home/skydev/restaurant_companion
# The following command essentially does: cd /home/skydev/restaurant_companion/scripts/..

cd "$(dirname "$0")/.."

python3 -m venv .venv


#********************************************************************************************************
# Special directive comment:  # shellcheck disable=SC1091
#    It is used to tell ShellCheck (a static analysis and linting tool for shell scripts) 
#    to # ignore error or warning SC1091 on the lines that follow.

  # shellcheck disable=SC1091


#********************************************************************************************************
# Command source (or its symbol equivalent .):
#    a built-in shell command that run a file directly inside your current terminal session.
#    e.g., source setup.sh 
#          . setup.sh

source .venv/bin/activate

pip install --upgrade pip

#*********************************************************************************************************
# Shell test command: [[ ... ]] (or old version [ ... ])
#    evaluate a conditional expression in side brackets
#    e.g., [[ $var == "" ]]  evaluates to be True if $var is empty or otherwise False.
# 

#*********************************************************************************************************
# Shell test command operators:
#  File test operators:
#   -f: File exists and is a regular file (not a directory), e.g., [[ -f setup.sh ]]
#   -e: File exists (any type: folder, file, link), e.g., [[ -e pyproject.toml ]]
#   -d: File exists and is a directory, e.g., [[ -d /var/log ]]
#   -s: File exists and is not empty (size > 0 bytes), e.g., [[ -s output.txt ]]
#
#  String test operators:
#   -z: String is empty (zero length), e.g., [[ -z "$username" ]]
#   -n: String is not empty (non-zero length), e.g., [[ -n "$username" ]]
#   =~: String matches a Regex (Only works in [[ ... ]]), e.g., [[ "$email" =~ @ ]]
#   = (or ==): Strings are exactly equal, e.g., [[ "$status" = "active" ]]
#   !=: Strings are not equal. e.g., [[ "$status" = "active" ]]
#
#  Integer (Numerical) Operators
#   -eq: Equal to, e.g., [[ "$count" -eq 10 ]]
#   -ne: Not equal to, e.g., [[ "$status_code" -ne 0 ]]
#   -gt: Greater than
#   -ge: Greater than or equal to
#   -lt: Less than
#   -le: Less than or equal to
#
#  Logical Operators
#   &&: AND, e.g., [[ "$A" == "1" && "$B" == "2" ]] 
#   ||: OR, e.g., [[ "$A" == "1" || "$B" == "2" ]]
#    !: NOT, e.g, [[ ! -f file ]]


#*********************************************************************************************************
# Arithmetic evaluation command: ((...)) or command expr
#   perform standard algebra, use regular math operators (+, -, *, /)
#   e.g., ((count = count + 1)) or count=$(expr $count + 1) or count=`expr $count + 1`
#


#*********************************************************************************************************
# Shell control structure:
#
#   if statement: conditional statement 
#       e.g., if [[ $status == "active" ]]; then
#                echo "System is running."
#             elif [[ $status == "maintenance" ]]; then
#                echo "System is in maintenance."
#             else
#                echo "System is offline."
#             fi
#
#   for loop: e.g., for user in alice bob charlie; do
#                       echo "Hello $user"
#                   done
#                
#                   for i in {1..5}; do
#                       echo "Count: $i"
#                   done 
#
#   while loop: e.g., count=1
#                     while [ "$count" -le 3 ]; do
#                       echo "Iteration $count"
#                       count=$((count + 1))
#                     done
#
#   until loop: e.g., count=1
#                     until [[ $count -gt 3 ]]; do
#                       echo "Iteration $count"
#                       count=$((count + 1))
#                     done
#


# ${}:  In Bash scripting, ${} represents parameter expansion, depending on syntax, it serves as:
#       1. Separating a variable from surrounding text: 
#          e.g., WORD="micro"; echo "${WORD}wave" # Outputs: microwave
#       2. Advanced manipulations:
#          e.g., echo "${VAR:-default_value}"  # Uses "default_value" if $VAR is empty
#       3. Substring extraction:
#          e.g., STRING="ABCDEF"; echo "${STRING:2}"   # Outputs: CDEF (Skips the first 2 characters)
#
# ${1}: It is the same as $1, represents the first positional argument passed to the script
#       Syntax ${1} is safer, e.g., 10th positional argment has to be ${10} 
#       beause $10 is intepreted by bash as $1 with a literal character 0 appended to the end.
# ${1:-}: If the first argument of the script is empty or missing, default it to an empty string.
#
if [[ "${1:-}" == "--all" ]]; then
  pip install -e ".[all,dev]"
else
  pip install -e ".[dev]"
fi

# chmod: change file permissions. Permision in Numeric Value: Read(r) = 4, Write(w) = 2, Execute(x) = 1
#        Numeric notation chmod 600 is equivalent to symbolic notation chmod u=rw,go=
#        add permission example in symbolic notation:    chmod ugo+rx
#        remove permission example in symbolic notation: chmod go-w
#        assign permission example in symbolic notation: chmod ug=rwx,o=r
#
if [[ ! -f .env ]]; then
  cp .env.example .env
  chmod 600 .env
  echo "Created .env - open it and paste your DEEPSEEK_API_KEY."
fi

echo
echo "Done. Next:  source .venv/bin/activate && pytest"
