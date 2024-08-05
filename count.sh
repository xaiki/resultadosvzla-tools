#!/bin/sh

maxprocs=16

mkdir -p failed

echo "Archivo,Acta,Nulos,Vacios,Maduro,Martinez,Bertucci,Brito,Ecarri,Fermin,Ceballos,Gonzalez,Marquez,Rausseo"
ls $1 | grep '.jpg' | xargs -P $maxprocs -n1 bash barimg.sh $1

