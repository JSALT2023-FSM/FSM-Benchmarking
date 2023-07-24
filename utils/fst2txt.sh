# for each file in folder create new file with same basename and .txt extension is not already exist

for file in *.fst; do
    if [ ! -f "${file%.fst}.txt" ]; then
        echo "$file"
        fstprint $file > ${file%.fst}.txt
    fi
done