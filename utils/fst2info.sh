for file in *.fst; do
    if [ ! -f "${file%.fst}.info" ]; then
        echo "$file"
        fstinfo $file > ${file%.fst}.info
    fi
done