#!/bin/bash

for filepath in "$@"; do
  filename=$(basename "$filepath")
  etag=$(curl -I "https://static.resultadosconvzla.com/$filename" | grep etag | sed -E 's/.*?([0-9a-f]{32}).*/\1/')
  echo "$etag $filename" | md5sum -c
done
