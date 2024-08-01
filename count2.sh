#!/bin/sh
echo "Acta,Maduro,Martinez,Bertucci,Brito,Ecarri,Fermin,Ceballos,Gonzalez,Marquez,Rausseo"
ls $1 | grep '.jpg' | xargs --max-procs=32 -n1 sh barimg.sh $1
