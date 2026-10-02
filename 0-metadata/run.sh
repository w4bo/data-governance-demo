#!/bin/bash
set -exo
cd /home/0-metadata

mkdir -p imgs
rm -f imgs/*.jpg imgs/*.jpeg imgs/*.JPG imgs/*.png imgs/SOURCES.tsv

download_image() {
  local filename="$1"
  local url="$2"
  local source="$3"

  curl --fail --location --retry 3 --connect-timeout 15 "$url" --output "imgs/${filename}"
}

download_image "cat.jpg" "https://commons.wikimedia.org/wiki/Special:FilePath/Cat-animal.jpg?width=800" "https://commons.wikimedia.org/wiki/File:Cat-animal.jpg"
download_image "dog.jpg" "https://commons.wikimedia.org/wiki/Special:FilePath/A_Dog.jpg?width=800" "https://commons.wikimedia.org/wiki/File:A_Dog.jpg"
download_image "elephant.jpg" "https://commons.wikimedia.org/wiki/Special:FilePath/Elephant_%281%29.jpg?width=800" "https://commons.wikimedia.org/wiki/File:Elephant_(1).jpg"
download_image "lion.jpg" "https://commons.wikimedia.org/wiki/Special:FilePath/Lion_%281%29.jpg?width=800" "https://commons.wikimedia.org/wiki/File:Lion_(1).jpg"
download_image "giraffe.jpg" "https://commons.wikimedia.org/wiki/Special:FilePath/Giraffe_%281%29.jpg?width=800" "https://commons.wikimedia.org/wiki/File:Giraffe_(1).jpg"
download_image "zebra.jpg" "https://commons.wikimedia.org/wiki/Special:FilePath/Etosha_Zebra.jpg?width=800" "https://commons.wikimedia.org/wiki/File:Etosha_Zebra.jpg"
download_image "tiger.jpg" "https://commons.wikimedia.org/wiki/Special:FilePath/Tiger%282%29.JPG?width=800" "https://commons.wikimedia.org/wiki/File:Tiger(2).JPG"
download_image "panda.jpg" "https://commons.wikimedia.org/wiki/Special:FilePath/Panda.jpg?width=800" "https://commons.wikimedia.org/wiki/File:Panda.jpg"
download_image "penguin.jpg" "https://commons.wikimedia.org/wiki/Special:FilePath/Gfp-penguin.jpg?width=800" "https://commons.wikimedia.org/wiki/File:Gfp-penguin.jpg"
download_image "kangaroo.jpg" "https://commons.wikimedia.org/wiki/Special:FilePath/Kangaroo.jpg?width=800" "https://commons.wikimedia.org/wiki/File:Kangaroo.jpg"

rm -rf out || true
python3.13 detect.py --input-dir imgs/ --output-dir ./out
python3.13 upload.py
