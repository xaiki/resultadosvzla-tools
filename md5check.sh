#!/bin/bash

for filename in $@; do
  etag=$(curl -I "https://static.resultadosconvzla.com/$filename" | grep etag | sed -E 's/.*?([0-9a-f]{32}).*/\1/')
  echo "$etag $filepath" | md5sum -c || echo "$etag - $filepath"
done
