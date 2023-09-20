#!/bin/bash

pid=$1
command=$2

# An entry in /proc means that the process is still running.
while [ -d "/proc/$pid" ]; do
    sleep 600
done

bash -c "$command"