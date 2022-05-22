#!/usr/bin/bash

SRC=$1
shift
TARGETS=$@

CHECKED_FILES=
CHECKED_FILES_FULL=

for TARGET in $TARGETS; do
    for FILE in $SRC/$TARGET/*.c; do
        ABS_FILE=`realpath $FILE`
        BASE_NAME=`basename $ABS_FILE`
        FILE_EXISTED=0
        for EXISTED_BASE_NAME in $CHECKED_FILES; do
            if [ "$EXISTED_BASE_NAME" == "$BASE_NAME" ]; then
                FILE_EXISTED=1
            fi
        done
        if [ "$FILE_EXISTED" == "0" ]; then
            CHECKED_FILES="$CHECKED_FILES $BASE_NAME"
            CHECKED_FILES_FULL="$CHECKED_FILES_FULL $ABS_FILE"
        fi
    done 
done

echo $CHECKED_FILES_FULL