#!/usr/bin/env bash
set -exo

find . -type f -name "*.sh" -exec chmod +x {} \;

ip=$(hostname -I | awk '{print $1}')

if [ -z "$ip" ]; then
  echo "Could not determine IP address."
  exit 1
fi

echo "Detected IP: $ip"
sed "s/MYIP/$ip/g" 2-catalog/ingest-config.example.yml > 2-catalog/ingest-config.yml
sed "s/MYIP/\"$ip\"/g" .env.example > .env
