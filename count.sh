#!/bin/bash
echo "record,maduro,gonzalez"
for filename in $1/*.jpg; do
  zbarimg -q $filename | awk '{ FS="!"; ORS=""; $0=$0; print substr($1,9,length($1)) "," } {FS=","; $0=$2; print $1+$2+$3+$4+$5+$6+$7+$8+$9+$10+$11+$12+$13 "," $34+$35+$36 "\n"}'
done
