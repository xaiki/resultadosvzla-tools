#!/bin/sh

grep -re 'static' resultadosconvzla.com/mesa/ | cut -d\" -f2 | sort -u | while read u; sh wget.sh -rc $u &; end 
