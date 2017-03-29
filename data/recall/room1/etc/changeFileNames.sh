#!/bin/bash
for d in */; do
 for f in $d*; do
   dir=$(echo $d | sed 's;/;;')
   target=$(echo $f | sed 's;audio;'"$dir"';')
   mv $f $target
 done
 cp $d/* .
done
