#!/bin/sh
echo "record,maduro,gonzalez"
ls $1 | grep '.jpg' | xargs --max-procs=32 -n1 sh barimg.sh $1
