#!/bin/sh
echo "Acta,Nulos,Vacios,Maduro,Martinez,Bertucci,Brito,Ecarri,Fermin,Ceballos,Gonzalez,Marquez,Rausseo,Nombre"
ls $1 | grep '.jpg' | xargs -P 32 -n 1 sh barimg.sh $1
